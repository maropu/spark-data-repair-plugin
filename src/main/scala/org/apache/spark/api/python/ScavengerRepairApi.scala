/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
v* this work for additional information regarding copyright ownership.
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
import org.apache.spark.util.{Utils => SparkUtils}

/** A Python API entry point for data cleaning. */
object ScavengerRepairApi extends BaseScavengerRepairApi {

  def injectNullAt(dbName: String, tableName: String, targetAttrList: String, nullRatio: Double): String = {
    logBasedOnLevel(s"injectNullAt called with: dbName=$dbName tableName=$tableName " +
      s"targetAttrList=$targetAttrList, nullRatio=$nullRatio")

    val df = withSparkSession { _ =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName)
      val targetAttrSet = if (targetAttrList.nonEmpty) {
        val attrSet = SparkUtils.stringToSeq(targetAttrList).toSet
        if (!tableAttrs.exists(attrSet.contains)) {
          throw new SparkException(s"No target attribute selected in $inputName")
        }
        attrSet
      } else {
        tableAttrs.toSet
      }
      val exprs = inputDf.schema.map {
        case f if targetAttrSet.contains(f.name) =>
          s"IF(rand() > $nullRatio, ${f.name}, NULL) AS ${f.name}"
        case f =>
          f.name
      }
      inputDf.selectExpr(exprs: _*)
    }
    Seq("injected" -> createAndCacheTempView(df)).asJson
  }

  /**
   * To compare result rows easily, this method flattens an input table
   * as a schema (`rowId`, attribute, val).
   */
  def flattenTable(dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"flattenTable called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    val df = withSparkSession { _ =>
      val (inputDf, _, _) = checkInputTable(dbName, tableName, rowId)
      val expr = inputDf.schema.filter(_.name != rowId)
        .map { f => s"STRUCT($rowId, '${f.name}', CAST(${f.name} AS STRING))" }
        .mkString("ARRAY(", ", ", ")")
      inputDf.selectExpr(s"INLINE($expr) AS (tid, attribute, val)")
    }
    Seq("flatten" -> createAndCacheTempView(df)).asJson
  }

  private def distinctStatMap(inputName: String): Map[String, Long] = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val df = spark.table(inputName)
    val tableStats = {
      val tableNode = df.queryExecution.analyzed.collectLeaves()
        .head.asInstanceOf[LeafNode]
      tableNode.computeStats()
    }
    val statMap = tableStats.attributeStats.map {
      kv => (kv._1.name, kv._2.distinctCount.map(_.toLong).get)
    }
    if (!df.columns.forall(statMap.contains)) {
      throw new SparkException(s"'$inputName' should be analyzed first")
    }
    statMap
  }

  private def computeAndGetTableStats(tableIdentifier: String): Map[String, Long] = {
    assert(SparkSession.getActiveSession.isDefined)
    val spark = SparkSession.getActiveSession.get
    // To compute the number of distinct values, runs `ANALYZE TABLE` first
    spark.sql(
      s"""
         |ANALYZE TABLE $tableIdentifier COMPUTE STATISTICS
         |FOR ALL COLUMNS
       """.stripMargin)

    distinctStatMap(tableIdentifier)
  }

  def computeDomainSizes(discreteAttrView: String): String = {
    logBasedOnLevel(s"computeDomainSizes called with: discreteAttrView=$discreteAttrView")

    withSparkSession { _ =>
      val statMap = computeAndGetTableStats(discreteAttrView)
      Seq("distinct_stats" -> statMap.mapValues(_.toString)).asJson
    }
  }

  def convertToDiscreteAttrs(
      dbName: String,
      tableName: String,
      rowId: String,
      discreteThres: Int,
      // TODO: Discretizes continuous values into buckets
      cntValueDiscretization: Boolean,
      blackAttrList: String): String = {

    logBasedOnLevel(s"convertToDiscreteAttrs called with: dbName=$dbName tableName=$tableName " +
      s"rowId=$rowId discreteThres=$discreteThres " +
      s"blackAttrList=${if (!blackAttrList.isEmpty) blackAttrList else "<none>"}")

    withSparkSession { _ =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName, rowId, blackAttrList)

      val statMap = computeAndGetTableStats(inputName)
      val attrsWithStats = tableAttrs.map { a => (a, statMap(a)) }
      val rowCnt = statMap(rowId)

      logBasedOnLevel({
        val distinctCnts = attrsWithStats.map { case (a, c) => s"distinctCnt($a):$c" }
        s"rowCnt:$rowCnt ${distinctCnts.mkString(" ")}"
      })

      if (attrsWithStats.collectFirst { case (a, cnt) if a == rowId => cnt }.get != rowCnt) {
        throw new SparkException(s"Uniqueness does not hold in column '$rowId' of table '$dbName.$tableName'.")
      }

      def isDiscrete(v: Long) = 1 < v && v < discreteThres
      val (discreteAttrs, nonDiscreteAttrs) = attrsWithStats.sortBy(_._2).partition { a => isDiscrete(a._2) }
      if (nonDiscreteAttrs.size > 1) {
        val droppedCols = nonDiscreteAttrs.filterNot(_._1 == rowId).map { case (a, c) => s"$a($c)" }
        logWarning("Dropped the columns having non-suitable domain size: " +
          droppedCols.mkString(", "))
      }
      if (discreteAttrs.size < 2) {
        val errMsg = if (discreteAttrs.nonEmpty) {
          s"found: ${discreteAttrs.map(_._1).mkString(",")}"
        } else {
          "no discrete attribute found."
        }
        throw new SparkException(s"$inputName must have more than one discrete attributes, but $errMsg")
      }

      // TODO: We could change this value to 64 in v3.1 (SPARK-30279) and we need to
      // support the more number of attributes for repair.
      val maxAttrNumToRepair = 32
      val attrSet = (if (discreteAttrs.size >= maxAttrNumToRepair) {
        val droppedCols = discreteAttrs.drop(maxAttrNumToRepair).map { case (a, c) => s"$a($c)" }
        logWarning(s"Maximum number of attributes is $maxAttrNumToRepair but " +
          s"the ${discreteAttrs.size} discrete attributes found in table '$dbName.$tableName', so " +
          s"the ${droppedCols.size} attributes dropped: ${droppedCols.mkString(",")}")
        discreteAttrs.take(maxAttrNumToRepair)
      } else {
        discreteAttrs
      }).map(_._1).toSet

      val discreteDf = inputDf.selectExpr(inputDf.columns.filter(attrSet.contains) :+ rowId: _*)
      Seq("discrete_attrs" -> createAndCacheTempView(discreteDf, "discrete_attrs"),
        "distinct_stats" -> statMap.mapValues(_.toString)).asJson
    }
  }

  def repairAttrsFrom(repairedCells: String, dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"repairAttrsFrom called with: repairedCellTable=$repairedCells " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      // `repairedCells` must have `$rowId`, `attribute`, and `repaired` columns
      checkIfColumnsExistIn(repairedCells, rowId :: "attribute" :: "val" :: Nil)

      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName, rowId)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $repairedCells")
          .collect.head.getSeq[String](0).toSet
      }

      val repairDf = sparkSession.sql(
        s"""
           |SELECT
           |  $rowId, map_from_entries(COLLECT_LIST(r)) AS repairs
           |FROM (
           |  select $rowId, struct(attribute, val) r
           |  FROM $repairedCells
           |)
           |GROUP BY
           |  $rowId
         """.stripMargin)

      val repaired = withTempView(inputDf) { inputView =>
        withTempView(repairDf) { repairView =>
          val cleanAttrs = tableAttrs.map {
            case attr if attr == rowId =>
              s"$inputView.$rowId"
            case attr if attrsToRepair.contains(attr) =>
              s"IF(ISNOTNULL(repairs['$attr']), repairs['$attr'], $attr) AS $attr"
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
      val tableAttrs = sparkSession.table(discreteAttrView).schema.map(_.name)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        tableAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
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
               |SELECT ${tableAttrs.mkString(", ")}, COUNT(1) cnt
               |FROM $inputView
               |GROUP BY GROUPING SETS (
               |  ${tableAttrs.map(a => s"($a)").mkString(", ")},
               |  ${pairSets.map(_.toSeq).map { case Seq(a1, a2) => s"($a1,$a2)" }.mkString(", ")}
               |)
               |$filterClauseOption
             """.stripMargin)
        }
      }
      Seq("attr_stats" -> createAndCacheTempView(statDf, "attr_stats")).asJson
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

  def computeDomainInErrorCells(
      discreteAttrView: String,
      attrStatView: String,
      errCellView: String,
      rowId: String,
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
      s"maxAttrsToComputeDomains=$maxAttrsToComputeDomains minCorrThres=$minCorrThres " +
      s"domain_threshold=alpha:$domain_threshold_alpha,beta=$domain_threshold_beta")

    withSparkSession { sparkSession =>
      val discreteAttrs = sparkSession.table(discreteAttrView).schema.map(_.name).filter(_ != rowId)
      val rowCnt = sparkSession.table(discreteAttrView).count()

      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0)
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
      val domainStatMap = distinctStatMap(discreteAttrView)

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
           |  extractField(struct(${cellExprs.mkString(", ")}), attribute) initValue
           |  $corrCols
           |FROM
           |  $discreteAttrView l, $errCellView r
           |WHERE
           |  l.$rowId = r.$rowId
         """.stripMargin)

      val domainInitValue = "CAST(NULL AS ARRAY<STRUCT<n: STRING, cnt: DOUBLE>>)"
      val cellDomainDf = if (domain_threshold_beta >= 1.0) {
        // The case where we don't need to compute error domains
        rvDf.selectExpr(rowId, "attribute", "initValue", s"$domainInitValue domain")
      } else {
        withTempView(rvDf) { rvView =>
          corrAttrs.map { case (attribute, corrAttrsWithScores) =>
            // Adds an empty domain for initial state
            val initDomainDf = sparkSession.sql(
              s"""
                 |SELECT $rowId, attribute, initValue, $domainInitValue domain $corrCols
                 |FROM $rvView
                 |WHERE attribute = '$attribute'
             """.stripMargin)

            val domainDf = if (corrAttrsWithScores.nonEmpty) {
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
                       |  initValue,
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
                   |  $rowId, attribute, initValue, domValue, SUM(score) score
                   |FROM (
                   |  SELECT
                   |    $rowId,
                   |    attribute,
                   |    initValue,
                   |    domValueWithFreq.n domValue,
                   |    exp(ln(cnt / $rowCnt) + ln(domValueWithFreq.cnt / cnt)) score
                   |  FROM (
                   |    SELECT
                   |      $rowId,
                   |      attribute,
                   |      initValue,
                   |      explode_outer(domain) domValueWithFreq
                   |    FROM
                   |      $domainView
                   |  ) d LEFT OUTER JOIN (
                   |    SELECT $attribute, MAX(cnt) cnt
                   |    FROM $attrStatView
                   |    WHERE ${whereCaluseToFilterStat(attribute)}
                   |    GROUP BY $attribute
                   |  ) s
                   |  ON
                   |    d.domValueWithFreq.n = s.$attribute
                   |)
                   |GROUP BY
                   |  $rowId, attribute, initValue, domValue
               """.stripMargin)
            }

            withTempView(domainWithScoreDf) { domainWithScoreView =>
              sparkSession.sql(
                s"""
                   |SELECT
                   |  l.$rowId,
                   |  l.attribute,
                   |  initValue,
                   |  filter(collect_set(named_struct('n', domValue, 'prob', score / denom)), x -> x.prob > $domain_threshold_beta) domain
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
                   |  l.$rowId, l.attribute, initValue
               """.stripMargin)
            }
          }
        }.reduce(_.union(_))
      }

      val cellDomainView = withJobDescription("compute domain values with posteriori probability") {
        createAndCacheTempView(cellDomainDf, "cell_domain")
      }
      // Number of rows in `cellDomainView` is the same with the number of error cells
      assert(cellDomainDf.count == sparkSession.table(errCellView).count)
      Seq("cell_domain" -> cellDomainView,
        "pairwise_attr_stats" -> corrAttrs).asJson
    }
  }
}
