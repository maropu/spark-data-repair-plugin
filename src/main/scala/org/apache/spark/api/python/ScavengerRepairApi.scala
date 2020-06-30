/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.api.python

import org.apache.spark.SparkException
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types._
import org.apache.spark.util.{Utils => SparkUtils}

/** A Python API entry point for data cleaning. */
object ScavengerRepairApi extends BaseScavengerRepairApi {

  private val continousTypes: Set[DataType] = Set(FloatType, DoubleType)
  private val supportedType: Set[DataType] = Set(StringType, BooleanType, ByteType, ShortType,
    IntegerType, LongType, DateType, TimestampType) ++ continousTypes

  def checkInputTable(dbName: String, tableName: String, rowId: String): String = {
    val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
    val unsupportedTypes = inputDf.schema.map(_.dataType).filterNot(supportedType.contains)
    if (unsupportedTypes.nonEmpty) {
      throw new SparkException(
        s"Supported types are ${supportedType.map(_.catalogString).mkString(",")}, but " +
          s"unsupported ones found: ${unsupportedTypes.map(_.catalogString).mkString(",")}")
    }
    val continousAttrs = inputDf.schema
      .filter(f => continousTypes.contains(f.dataType)).map(_.name).mkString(",")
    Seq("input_table" -> qualifiedName,
      "num_input_rows" -> s"${inputDf.count}",
      "num_attrs" -> s"${inputDf.columns.length}",
      "continous_attrs" -> continousAttrs
    ).asJson
  }

  private case class ColumnStat(distinctCount: Long, min: Option[Any], max: Option[Any])

  private def getColumnStats(inputName: String): Map[String, ColumnStat] = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val df = spark.table(inputName)
    val tableStats = {
      val tableNode = df.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
      tableNode.computeStats()
    }
    val statMap = tableStats.attributeStats.map { kv =>
      val stat = kv._2
      val distinctCount = stat.distinctCount.map(_.toLong)
      (kv._1.name, ColumnStat(distinctCount.get, stat.min, stat.max))
    }
    // assert(df.columns.toSet == statMap.keySet)
    statMap
  }

  private def computeAndGetTableStats(tableIdentifier: String): Map[String, ColumnStat] = {
    assert(SparkSession.getActiveSession.isDefined)
    val spark = SparkSession.getActiveSession.get
    // For safe guards, just cache it for `ANALYZE TABLE`
    spark.table(tableIdentifier).cache()
    spark.sql(
      s"""
         |ANALYZE TABLE $tableIdentifier COMPUTE STATISTICS
         |FOR ALL COLUMNS
       """.stripMargin)

    getColumnStats(tableIdentifier)
  }

  def computeDomainSizes(discreteAttrView: String): String = {
    logBasedOnLevel(s"computeDomainSizes called with: discreteAttrView=$discreteAttrView")

    withSparkSession { _ =>
      val statMap = computeAndGetTableStats(discreteAttrView)
      Seq("distinct_stats" -> statMap.mapValues(_.distinctCount.toString)).asJson
    }
  }

  def convertToDiscreteFeatures(
      dbName: String,
      tableName: String,
      rowId: String,
      discreteThres: Int): String = {

    logBasedOnLevel(s"convertToDiscreteFeatures called with: dbName=$dbName tableName=$tableName " +
      s"rowId=$rowId discreteThres=$discreteThres")

    require(rowId.nonEmpty, s"$rowId should be a non-empty string.")
    require(2 <= discreteThres && discreteThres < 65536, "discreteThres should be in [2, 65536).")

    withSparkSession { _ =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
      val statMap = computeAndGetTableStats(qualifiedName)
      val attrTypeMap = inputDf.schema.map { f => f.name -> f.dataType }.toMap
      val rowCnt = inputDf.count()
      if (inputDf.selectExpr(rowId).distinct().count() != rowCnt) {
        throw new SparkException(s"Uniqueness does not hold in column '$rowId' " +
          s"of table '$dbName.$tableName'.")
      }
      val discreteExprs = inputDf.columns.flatMap { attr =>
        (statMap(attr), attrTypeMap(attr)) match {
          case (ColumnStat(_, min, max), tpe) if continousTypes.contains(tpe) =>
            logBasedOnLevel(s"'$attr' regraded as a continuous attribute (min=${min.get}, " +
              s"max=${max.get}), so discretized into [0, $discreteThres)")
            Some(s"int(($attr - ${min.get}) / (${max.get} - ${min.get}) * $discreteThres) $attr")
          case (ColumnStat(distinctCnt, _, _), _)
              if attr == rowId || (1 < distinctCnt && distinctCnt < discreteThres) =>
            Some(attr)
          case (ColumnStat(distinctCnt, _, _), _) =>
            logWarning(s"'$attr' dropped because of its unsuitable domain (size=$distinctCnt)")
            None
        }
      }
      val discreteDf = inputDf.selectExpr(discreteExprs: _*)
      val distinctStats = statMap.mapValues(_.distinctCount.toString)
      Seq("discrete_features" -> createAndCacheTempView(discreteDf, "discrete_features"),
        "distinct_stats" -> distinctStats
      ).asJson
    }
  }

  def repairAttrsFrom(repairedCells: String, dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"repairAttrsFrom called with: repairedCellTable=$repairedCells " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      // `repairedCells` must have `$rowId`, `attribute`, and `repaired` columns
      checkIfColumnsExistIn(repairedCells, rowId :: "attribute" :: "value" :: Nil)

      val (inputDf, _) = checkAndGetInputTable(dbName, tableName, rowId)
      val continousAttrTypeMap = inputDf.schema.filter(f => continousTypes.contains(f.dataType))
        .map { f => f.name -> f.dataType }.toMap
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $repairedCells")
          .collect.head.getSeq[String](0).toSet
      }

      val repairDf = sparkSession.sql(
        s"""
           |SELECT
           |  $rowId, map_from_entries(COLLECT_LIST(r)) AS repairs
           |FROM (
           |  select $rowId, struct(attribute, value) r
           |  FROM $repairedCells
           |)
           |GROUP BY
           |  $rowId
         """.stripMargin)

      val repaired = withTempView(inputDf) { inputView =>
        withTempView(repairDf) { repairView =>
          val cleanAttrs = inputDf.columns.map {
            case attr if attr == rowId =>
              s"$inputView.$rowId"
            case attr if attrsToRepair.contains(attr) =>
              val repaired = if (continousAttrTypeMap.contains(attr)) {
                val dataType = continousAttrTypeMap(attr)
                s"CAST(repairs['$attr'] AS ${dataType.catalogString})"
              } else {
                s"repairs['$attr']"
              }
              s"IF(ISNOTNULL(repairs['$attr']), $repaired, $attr) AS $attr"
            case cleanAttr =>
              cleanAttr
          }
          sparkSession.sql(
            s"""
               |SELECT ${cleanAttrs.mkString(", ")}
               |FROM $inputView
               |LEFT OUTER JOIN $repairView
               |ON $inputView.$rowId = $repairView.$rowId
             """.stripMargin)
        }
      }
      Seq("repaired" -> createAndCacheTempView(repaired, "repaired")).asJson
    }
  }

  def convertErrorCellsToNull(discreteAttrView: String, errCellView: String, rowId: String): String = {
    logBasedOnLevel(s"convertErrorCellsToNull called with: discreteAttrView=$discreteAttrView " +
      s"errCellView=$errCellView rowId=$rowId")

    withSparkSession { sparkSession =>
      // `errCellView` must have `$rowId` and `attribute` columns
      checkIfColumnsExistIn(errCellView, rowId :: "attribute" :: Nil)

      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0).toSet
      }

      val errAttrDf = sparkSession.sql(
        s"""
           |SELECT $rowId, collect_set(attribute) AS errors
           |FROM $errCellView
           |GROUP BY $rowId
         """.stripMargin)

      val repairBase = withTempView(errAttrDf) { errAttrView =>
        val cleanAttrs = sparkSession.table(discreteAttrView).columns.map {
          case attr if attr == rowId =>
            s"$discreteAttrView.$rowId"
          case attr if attrsToRepair.contains(attr) =>
            s"IF(array_contains(errors, '$attr'), NULL, $attr) AS $attr"
          case cleanAttr =>
            cleanAttr
        }
        sparkSession.sql(
          s"""
             |SELECT ${cleanAttrs.mkString(", ")}
             |FROM $discreteAttrView
             |LEFT OUTER JOIN $errAttrView
             |ON $discreteAttrView.$rowId = $errAttrView.$rowId
           """.stripMargin)
      }
      Seq("repair_base" -> createAndCacheTempView(repairBase, "repair_base")).asJson
    }
  }

  def computeAttrStats(
      discreteAttrView: String,
      errCellView: String,
      rowId: String,
      sampleRatio: Double,
      statThresRatio: Double): String = {

    logBasedOnLevel(s"computeAttrStats called with: discreteAttrView=$discreteAttrView " +
      s"errCellView=$errCellView rowId=$rowId sampleRatio=$sampleRatio statThresRatio=$statThresRatio")

    withSparkSession { sparkSession =>
      // Computes numbers for single and pair-wise statistics in the input table
      val discreteAttrs = sparkSession.table(discreteAttrView).schema.map(_.name).filter(_ != rowId)
      val attrsToRepair = {
        val attrs = sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0)
        attrs.filter(discreteAttrs.contains)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      val statDf = {
        val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
        val inputDf = if (sampleRatio < 1.0) {
          sparkSession.table(discreteAttrView).sample(sampleRatio)
        } else {
          sparkSession.table(discreteAttrView)
        }
        withTempView(inputDf) { inputView =>
          val filterClauseOption = if (statThresRatio > 0.0) {
            val cond = s"HAVING cnt > ${(inputDf.count * statThresRatio).toInt}"
            logBasedOnLevel(s"Attributes stats filter enabled: $cond")
            cond
          } else {
            ""
          }
          sparkSession.sql(
            s"""
               |SELECT ${discreteAttrs.mkString(", ")}, COUNT(1) cnt
               |FROM $inputView
               |GROUP BY GROUPING SETS (
               |  ${discreteAttrs.map(a => s"($a)").mkString(", ")},
               |  ${pairSets.map(_.toSeq).map { case Seq(a1, a2) => s"($a1,$a2)" }.mkString(", ")}
               |)
               |$filterClauseOption
             """.stripMargin)
        }
      }
      Seq("attr_stats" -> createAndCacheTempView(statDf, "attr_stats")).asJson
    }
  }

  def computeDomainInErrorCells(
      discreteAttrView: String,
      attrStatView: String,
      errCellView: String,
      rowId: String,
      continuousAttrList: String,
      maxAttrsToComputeDomains: Int,
      minCorrThres: Double,
      domain_threshold_alpha: Double,
      domain_threshold_beta: Double): String = {

    require(0 < maxAttrsToComputeDomains, "maxAttrsToComputeDomains should be greater than 0.")
    require(0.0 <= minCorrThres && minCorrThres < 1.0, "minCorrThres should be in [0.0, 1.0).")
    require(0.0 <= domain_threshold_alpha && domain_threshold_alpha < 1.0,
      "domain_threashold_alpha should be in [0.0, 1.0).")
    require(0.0 <= domain_threshold_beta && domain_threshold_beta <= 1.0,
      "domain_threashold_beta should be in [0.0, 1.0].")

    logBasedOnLevel(s"computeDomainInErrorCells called with: discreteAttrView=$discreteAttrView " +
      s"attrStatView=$attrStatView errCellView=$errCellView rowId=$rowId " +
      s"continousAttrList=${if (!continuousAttrList.isEmpty) continuousAttrList else "<none>"} " +
      s"maxAttrsToComputeDomains=$maxAttrsToComputeDomains minCorrThres=$minCorrThres " +
      s"domain_threshold=alpha:$domain_threshold_alpha,beta=$domain_threshold_beta")

    withSparkSession { sparkSession =>
      val discreteAttrs = sparkSession.table(discreteAttrView).schema.map(_.name).filter(_ != rowId)
      val rowCnt = sparkSession.table(discreteAttrView).count()

      val attrsToRepair = {
        val attrs = sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0)
        attrs.filter(discreteAttrs.contains)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      def whereCaluseToFilterStat(a: String): String =
        s"$a IS NOT NULL AND ${discreteAttrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"

      // Computes the conditional entropy: H(x|y) = H(x,y) - H(y).
      // H(x,y) denotes H(x U y). If H(x|y) = 0, then y determines x, i.e., y -> x.
      val hXYs = withJobDescription("compute conditional entropy H(x|y)") {
        val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
        pairSets.map { attrPairKey =>
          attrPairKey -> {
            val Seq(a1, a2) = attrPairKey.toSeq
            val df = sparkSession.sql(
              s"""
                 |SELECT -SUM(hXY) hXY
                 |FROM (
                 |  SELECT $a1 X, $a2 Y, (cnt / $rowCnt) * log10(cnt / $rowCnt) hXY
                 |  FROM $attrStatView
                 |  WHERE $a1 IS NOT NULL AND
                 |    $a2 IS NOT NULL
                 |)
               """.stripMargin)

            df.collect().map { row =>
              if (!row.isNullAt(0)) row.getDouble(0) else 0.0
            }.head
          }
        }.toMap
      }

      val hYs = withJobDescription("compute entropy H(y)") {
        discreteAttrs.map { attrKey =>
          attrKey -> {
            val df = sparkSession.sql(
              s"""
                 |SELECT -SUM(hY) hY
                 |FROM (
                 |  /* TODO: Needs to reconsider how-to-handle NULL */
                 |  /* Use `MAX` to drop ($attrKey, null) tuples in `$discreteAttrView` */
                 |  SELECT $attrKey Y, (MAX(cnt) / $rowCnt) * log10(MAX(cnt) / $rowCnt) hY
                 |  FROM $attrStatView
                 |  WHERE ${whereCaluseToFilterStat(attrKey)}
                 |  GROUP BY $attrKey
                 |)
               """.stripMargin)

            df.collect().map { row =>
              if (!row.isNullAt(0)) row.getDouble(0) else 0.0
            }.head
          }
        }.toMap
      }

      // Uses the domain size of X as a log base for normalization
      val domainStatMap = computeAndGetTableStats(discreteAttrView).mapValues(_.distinctCount)

      val pairWiseStats = attrPairsToRepair.map { case attrPair @ (rvX, rvY) =>
        // The conditional entropy is 0 for strongly correlated attributes and 1 for completely independent
        // attributes. We reverse this to reflect the correlation.
        val domainSize = domainStatMap(rvX)
        attrPair -> (1.0 - ((hXYs(Set(rvX, rvY)) - hYs(rvY)) / scala.math.log10(domainSize)))
      }

      val pairWiseStatMap = pairWiseStats.groupBy { case ((attrToRepair, _), _) =>
        attrToRepair
      }.map { case (k, v) =>
        k -> v.map { case ((_, attr), v) =>
          (attr, v)
        }.sortBy(_._2).reverse
      }
      logBasedOnLevel({
        val pairStats = pairWiseStatMap.map { case (k, v) =>
          val stats = v.map { case (attribute, h) =>
            val isEmployed = if (h > minCorrThres) "*" else ""
            s"$isEmployed$attribute:$h"
          }.mkString("\n    ")
          s"""$k (min=${v.last._2} max=${v.head._2}):
             |    $stats
           """.stripMargin
        }
        s"""
           |Pair-wise statistics H(x,y) '*' means it exceeds the threshold(=$minCorrThres):
           |
           |  ${pairStats.mkString("\n  ")}
         """.stripMargin
      })

      val corrAttrs = pairWiseStatMap.map { case (k, v) =>
        val attrs = v.filter(_._2 > minCorrThres)
        (k, if (attrs.size > maxAttrsToComputeDomains) {
          attrs.take(maxAttrsToComputeDomains)
        } else if (attrs.isEmpty) {
          // If correlated attributes not found, we pick up data from its domain randomly
          logWarning(s"Correlated attributes not found for $k")
          Nil
        } else {
          attrs
        })
      }

      val attrToId = discreteAttrs.zipWithIndex.toMap
      sparkSession.udf.register("extractField", (row: Row, attribute: String) => {
        row.getString(attrToId(attribute))
      })
      val cellExprs = discreteAttrs.map { a => s"CAST(l.$a AS STRING) $a" }
      // Needs to keep the correlated attributes for selecting their domains
      val corrAttrSet = corrAttrs.flatMap(_._2.map(_._1)).toSet
      val corrCols = if (corrAttrSet.nonEmpty) {
        corrAttrSet.mkString(", ", ", ", "")
      } else {
        ""
      }
      val rvDf = sparkSession.sql(
        s"""
           |SELECT
           |  l.$rowId,
           |  attribute,
           |  extractField(struct(${cellExprs.mkString(", ")}), attribute) current_value
           |  $corrCols
           |FROM
           |  $discreteAttrView l, $errCellView r
           |WHERE
           |  l.$rowId = r.$rowId
         """.stripMargin)

      val domainInitValue = "CAST(NULL AS ARRAY<STRUCT<n: STRING, cnt: DOUBLE>>)"
      val cellDomainDf = if (domain_threshold_beta >= 1.0) {
        // The case where we don't need to compute error domains
        rvDf.selectExpr(rowId, "attribute", "current_value", s"$domainInitValue domain")
      } else {
        withTempView(rvDf) { rvView =>
          val continousAttrs = SparkUtils.stringToSeq(continuousAttrList).toSet
          corrAttrs.map { case (attribute, corrAttrsWithScores) =>
            // Adds an empty domain for initial state
            val initDomainDf = sparkSession.sql(
              s"""
                 |SELECT $rowId, attribute, current_value, $domainInitValue domain $corrCols
                 |FROM $rvView
                 |WHERE attribute = '$attribute'
               """.stripMargin)

            val domainDf = if (!continousAttrs.contains(attribute) &&
                corrAttrsWithScores.nonEmpty) {
              val corrAttrs = corrAttrsWithScores.map(_._1)
              logBasedOnLevel(s"Computing '$attribute' domain from ${corrAttrs.size} correlated " +
                s"attributes (${corrAttrs.mkString(",")})...")

              corrAttrs.foldLeft(initDomainDf) { case (df, attr) =>
                withTempView(df) { domainSpaceView =>
                  val tau = {
                    // `tau` becomes a threshold on co-occurrence frequency
                    val productSpaceSize = domainStatMap(attr) * domainStatMap(attribute)
                    (domain_threshold_alpha * (rowCnt / productSpaceSize)).toLong
                  }
                  sparkSession.sql(
                    s"""
                       |SELECT
                       |  $rowId,
                       |  attribute,
                       |  current_value,
                       |  IF(ISNOTNULL(l.domain), CONCAT(l.domain, r.d), r.d) domain,
                       |  ${corrAttrSet.map(a => s"l.$a").mkString(",")}
                       |FROM
                       |  $domainSpaceView l
                       |LEFT OUTER JOIN (
                       |  SELECT $attr, collect_set(named_struct('n', $attribute, 'cnt', array_max(array(double(cnt) - 1.0, 0.1)))) d
                       |  FROM (
                       |    SELECT *
                       |    FROM $attrStatView
                       |    WHERE $attribute IS NOT NULL AND
                       |      $attr IS NOT NULL AND
                       |      cnt > $tau
                       |  )
                       |  GROUP BY
                       |    $attr
                       |) r
                       |ON
                       |  l.$attr = r.$attr
                     """.stripMargin)
                }
              }
            } else {
              initDomainDf
            }

            // To prune the domain, we use NaiveBayes that is an estimator of posterior probabilities
            // using the naive independence assumption where
            //   p(v_cur | v_init) = p(v_cur) * \prod_i (v_init_i | v_cur)
            // where v_init_i is the init value for corresponding to attribute i.
            val domainWithScoreDf = withTempView(domainDf) { domainView =>
              sparkSession.sql(
                s"""
                   |SELECT
                   |  $rowId, attribute, current_value, domain_value, SUM(score) score
                   |FROM (
                   |  SELECT
                   |    $rowId,
                   |    attribute,
                   |    current_value,
                   |    domain_value_with_freq.n domain_value,
                   |    exp(ln(cnt / $rowCnt) + ln(domain_value_with_freq.cnt / cnt)) score
                   |  FROM (
                   |    SELECT
                   |      $rowId,
                   |      attribute,
                   |      current_value,
                   |      explode_outer(domain) domain_value_with_freq
                   |    FROM
                   |      $domainView
                   |  ) d LEFT OUTER JOIN (
                   |    SELECT $attribute, MAX(cnt) cnt
                   |    FROM $attrStatView
                   |    WHERE ${whereCaluseToFilterStat(attribute)}
                   |    GROUP BY $attribute
                   |  ) s
                   |  ON
                   |    d.domain_value_with_freq.n = s.$attribute
                   |)
                   |GROUP BY
                   |  $rowId, attribute, current_value, domain_value
                 """.stripMargin)
            }

            withTempView(domainWithScoreDf) { domainWithScoreView =>
              sparkSession.sql(
                s"""
                   |SELECT
                   |  l.$rowId,
                   |  l.attribute,
                   |  current_value,
                   |  filter(collect_set(named_struct('n', domain_value, 'prob', score / denom)), x -> x.prob > $domain_threshold_beta) domain
                   |FROM
                   |  $domainWithScoreView l, (
                   |    SELECT
                   |      $rowId, attribute, SUM(score) denom
                   |    FROM
                   |      $domainWithScoreView
                   |    GROUP BY
                   |      $rowId, attribute
                   |  ) r
                   |WHERE
                   |  l.$rowId = r.$rowId AND l.attribute = r.attribute
                   |GROUP BY
                   |  l.$rowId, l.attribute, current_value
                 """.stripMargin)
            }
          }
        }.reduce(_.union(_))
      }

      val cellDomainView = withJobDescription("compute domain values with posteriori probability") {
        createAndCacheTempView(cellDomainDf, "cell_domain")
      }
      // Number of rows in `cellDomainView` is the same with the number of error cells
      // assert(cellDomainDf.count == sparkSession.table(errCellView).count)
      Seq("cell_domain" -> cellDomainView,
        "pairwise_attr_stats" -> pairWiseStatMap.mapValues(seqToJson)
      ).asJson
    }
  }
}
