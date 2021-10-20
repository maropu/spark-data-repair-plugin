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

import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql._
import org.apache.spark.util.RepairUtils._
import org.apache.spark.util.{Utils => SparkUtils}

/** A Python API entry point for data cleaning. */
object RepairApi extends RepairBase {

  def checkInputTable(dbName: String, tableName: String, rowId: String): String = {
    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName(dbName, tableName, rowId)

    val unsupportedTypes = inputDf.schema.map(_.dataType).filterNot(supportedType.contains)
    if (unsupportedTypes.nonEmpty) {
      val supportedTypeMsg = supportedType.map(_.catalogString).mkString(",")
      val unsupportedTypeMsg = unsupportedTypes.map(_.catalogString).mkString(",")
      throw AnalysisException(s"Supported types are $supportedTypeMsg, but " +
        s"unsupported ones found: $unsupportedTypeMsg")
    }

    // Checks if the input table has enough columns for repairing
    if (!(inputDf.columns.length >= 3)) {
      throw AnalysisException(s"A least three columns (`$rowId` columns + two more ones) " +
        s"in table '$qualifiedName'")
    }

    // Checks if `row_id` is unique
    val rowCount = inputDf.count()
    val distinctCount = inputDf.selectExpr(rowId).distinct().count()
    if (distinctCount != rowCount) {
      throw AnalysisException(s"Uniqueness does not hold in column '$rowId' " +
        s"of table '$qualifiedName' (# of distinct '$rowId': $distinctCount, # of rows: $rowCount)")
    }

    val continousAttrs = inputDf.schema
      .filter(f => f.name != rowId && continousTypes.contains(f.dataType))
      .map(_.name).mkString(",")

    Seq("input_table" -> qualifiedName,
      "num_input_rows" -> s"${inputDf.count}",
      "num_attrs" -> s"${inputDf.columns.length - 1}",
      "continous_attrs" -> continousAttrs
    ).asJson
  }

  def withCurrentValues(
      inputView: String,
      errCellView: String,
      rowId: String,
      targetAttrList: String): DataFrame = {
    logBasedOnLevel(s"withCurrentValues called with: inputView=$inputView " +
      s"errCellView=$errCellView rowId=$rowId targetAttrList=$targetAttrList")

    assert(checkSchema(errCellView, "attribute STRING", rowId, strict = false))

    val targetAttrs = SparkUtils.stringToSeq(targetAttrList)
    assert(targetAttrs.nonEmpty)
    assert({
      val inputAttrSet = spark.table(inputView).columns.toSet
      targetAttrs.forall(inputAttrSet.contains)
    })

    val attrToId = targetAttrs.zipWithIndex.toMap
    spark.udf.register("extractField", (row: Row, attribute: String) => {
      row.getString(attrToId(attribute))
    })
    val cellExprs = targetAttrs.map { a => s"CAST(r.`$a` AS STRING) $a" }
    val df = spark.sql(
      s"""
         |SELECT
         |  l.$rowId,
         |  l.attribute,
         |  extractField(struct(${cellExprs.mkString(", ")}), l.attribute) current_value
         |FROM
         |  $errCellView l, $inputView r
         |WHERE
         |  l.$rowId = r.$rowId
       """.stripMargin)
    assert(checkSchema(df, "attribute STRING, current_value STRING", rowId, strict = true))
    df
  }

  case class ColumnStat(distinctCount: Long, min: Option[Any], max: Option[Any])

  private[python] def computeAndGetTableStats(tableIdent: String): Map[String, ColumnStat] = {
    val df = spark.table(tableIdent)
    val relation = df.queryExecution.analyzed
    val (_, colStats) = SparkCommandUtils.computeColumnStats(spark, relation, relation.output)
    val columnStats = colStats.map { case (attr, stat) =>
      val distinctCount = stat.distinctCount.map(_.toLong)
      (attr.name, ColumnStat(distinctCount.get, stat.min, stat.max))
    }
    assert(df.columns.forall(columnStats.contains))
    columnStats
  }

  def computeDomainSizes(discretizedInputView: String): String = {
    logBasedOnLevel(s"computeDomainSizes called with: discretizedInputView=$discretizedInputView")
    val statMap = computeAndGetTableStats(discretizedInputView)
    Seq("distinct_stats" -> statMap.mapValues(_.distinctCount.toString)).asJson
  }

  private def discretizeTable(
      inputView: String,
      discreteThres: Int,
      whitelist: Set[String] = Set.empty): (DataFrame, Map[String, ColumnStat]) = {
    assert(2 <= discreteThres && discreteThres < 65536, "discreteThres should be in [2, 65536).")
    val statMap = computeAndGetTableStats(inputView)
    val inputDf = spark.table(inputView)
    val attrTypeMap = inputDf.schema.map { f => f.name -> f.dataType }.toMap
    val discretizedExprs = inputDf.columns.flatMap { attr =>
      (statMap(attr), attrTypeMap(attr)) match {
        case (ColumnStat(_, min, max), tpe) if continousTypes.contains(tpe) =>
          logBasedOnLevel(s"'$attr' regraded as a continuous attribute (min=${min.get}, " +
            s"max=${max.get}), so discretized into [0, $discreteThres)")
          Some(s"int(($attr - ${min.get}) / (${max.get} - ${min.get}) * $discreteThres) $attr")
        case (ColumnStat(distinctCount, _, _), _)
          if whitelist.contains(attr) || (1 < distinctCount && distinctCount < discreteThres) =>
          Some(attr)
        case (ColumnStat(distinctCount, _, _), _) =>
          logWarning(s"'$attr' dropped because of its unsuitable domain (size=$distinctCount)")
          None
      }
    }
    val df = inputDf.selectExpr(discretizedExprs: _*)
    (df, statMap)
  }

  def convertToDiscretizedTable(
      qualifiedName: String,
      rowId: String,
      discreteThres: Int): String = {
    assert(rowId.nonEmpty, s"$rowId should be a non-empty string.")
    logBasedOnLevel(s"convertToDiscretizedTable called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId discreteThres=$discreteThres")
    val (discreteDf, statMap) = discretizeTable(qualifiedName, discreteThres, Set(rowId))
    val distinctStats = statMap.mapValues(_.distinctCount.toString)
    val discretizedView = createAndCacheTempView(discreteDf, "discretized_table")
    Seq("discretized_table" -> discretizedView,
      "distinct_stats" -> distinctStats
    ).asJson
  }

  def convertErrorCellsToNull(
      discretizedInputView: String,
      errCellView: String,
      rowId: String,
      targetAttrList: String): String = {
    logBasedOnLevel(s"convertErrorCellsToNull called with: discretizedInputView=$discretizedInputView " +
      s"errCellView=$errCellView rowId=$rowId targetAttrList=$targetAttrList")

    // `errCellView` schema must have `$rowId` and `attribute` columns
    assert(checkSchema(errCellView, "attribute STRING", rowId, strict = true))

    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)
    assert(attrsToRepair.nonEmpty)

    val errAttrDf = spark.sql(
      s"""
         |SELECT $rowId, collect_set(attribute) AS errors
         |FROM $errCellView
         |GROUP BY $rowId
       """.stripMargin)

    val repairBase = withTempView(errAttrDf) { errAttrView =>
      val cleanAttrs = spark.table(discretizedInputView).columns.map {
        case attr if attr == rowId =>
          s"$discretizedInputView.$rowId"
        case attr if attrsToRepair.contains(attr) =>
          s"IF(array_contains(errors, '$attr'), NULL, $attr) AS $attr"
        case cleanAttr =>
          cleanAttr
      }
      spark.sql(
        s"""
           |SELECT ${cleanAttrs.mkString(", ")}
           |FROM $discretizedInputView
           |LEFT OUTER JOIN $errAttrView
           |ON $discretizedInputView.$rowId = $errAttrView.$rowId
         """.stripMargin)
    }
    val repairBaseView = createAndCacheTempView(repairBase, "repair_base_cells")
    Seq("repair_base_cells" -> repairBaseView).asJson
  }

  def computeFunctionalDeps(inputView: String, constraintFilePath: String): String = {
    logBasedOnLevel(s"computeFunctionalDep called with: discretizedInputView=$inputView " +
      s"constraintFilePath=$constraintFilePath")
    DepGraph.computeFunctionalDeps(inputView, constraintFilePath)
  }

  def computeFunctionalDepMap(inputView: String, X: String, Y: String): String = {
    logBasedOnLevel(s"computeFunctionalDepMap called with: inputView=$inputView X=$X Y=$Y")
    DepGraph.computeFunctionalDepMap(inputView, X, Y)
  }

  private[python] def computeFreqStats(
      inputView: String,
      targetAttrSets: Seq[Seq[String]],
      statSampleRatio: Double,
      statThreshold: Double): DataFrame = {
    assert(targetAttrSets.nonEmpty)
    assert(0.0 <= statSampleRatio && statSampleRatio <= 1.0)
    assert(0.0 <= statThreshold && statThreshold <= 1.0)

    val targetAttrs = targetAttrSets.flatten.distinct
    val distinctTargetAttrSet = targetAttrSets.map(_.toSet).distinct.map(_.toSeq)
    // TODO: Needs to check the maximum size of grouping sets
    // assert(distinctTargetAttrSet.length < 64,
    //   "Cannot handle the target set whose size is more than 63")

    val groupingSetSeq = distinctTargetAttrSet.map {
      case Seq(a) => s"(`$a`)"
      case Seq(a1, a2) => s"(`$a1`,`$a2`)"
      case attrs =>
        throw new IllegalStateException(
          s"Cannot handle more than two entries: ${attrs.mkString(",")}")
    }
    val inputDf = if (statSampleRatio < 1.0) {
      spark.table(inputView).sample(statSampleRatio)
    } else {
      spark.table(inputView)
    }
    withTempView(inputDf) { inputView =>
      val filterClauseOption = if (statThreshold > 0.0) {
        val cond = s"HAVING cnt > ${(inputDf.count * statThreshold).toInt}"
        logBasedOnLevel(s"Attributes stats filter enabled: $cond")
        cond
      } else {
        ""
      }
      spark.sql(
        s"""
           |SELECT ${targetAttrs.mkString(", ")}, COUNT(1) cnt
           |FROM $inputView
           |GROUP BY GROUPING SETS (
           |  ${groupingSetSeq.mkString(", ")}
           |)
           |$filterClauseOption
         """.stripMargin)
    }
  }

  private def whereCaluseToFilterStat(a: String, attrs: Seq[String]): String = {
    s"$a IS NOT NULL AND ${attrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"
  }

  private[python] def computePairwiseStats(
       inputView: String,
       rowCount: Long,
       freqStatView: String,
       targetAttrs: Seq[String],
       targetAttrPairsToComputeStats: Seq[(String, String)],
       domainStatMap: Map[String, Long]): Map[String, Seq[(String, Double)]] = {
    assert(rowCount > 0)
    assert(targetAttrs.nonEmpty && targetAttrPairsToComputeStats.nonEmpty)
    assert(targetAttrs.forall(domainStatMap.contains))

    // Computes the conditional entropy: H(x|y) = H(x,y) - H(y).
    // H(x,y) denotes H(x U y). If H(x|y) = 0, then y determines x, i.e., y -> x.
    val hXYs = withJobDescription("compute conditional entropy H(x|y)") {
      val pairSets = targetAttrPairsToComputeStats.map(p => Set(p._1, p._2)).distinct
      pairSets.map { attrPairKey =>
        attrPairKey -> {
          val Seq(a1, a2) = attrPairKey.toSeq
          val df = spark.sql(
            s"""
               |SELECT -SUM(hXY) hXY
               |FROM (
               |  SELECT $a1 X, $a2 Y, (cnt / $rowCount) * log10(cnt / $rowCount) hXY
               |  FROM $freqStatView
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
      targetAttrs.map { attrKey =>
        attrKey -> {
          val df = spark.sql(
            s"""
               |SELECT -SUM(hY) hY
               |FROM (
               |  /* TODO: Needs to reconsider how-to-handle NULL */
               |  /* Use `MAX` to drop ($attrKey, null) tuples in `$inputView` */
               |  SELECT $attrKey Y, (MAX(cnt) / $rowCount) * log10(MAX(cnt) / $rowCount) hY
               |  FROM $freqStatView
               |  WHERE ${whereCaluseToFilterStat(attrKey, targetAttrs)}
               |  GROUP BY $attrKey
               |)
             """.stripMargin)

          df.collect().map { row =>
            if (!row.isNullAt(0)) {
              row.getDouble(0)
            } else {
              logWarning(s"No frequency stat found for $attrKey")
              0.0
            }
          }.head
        }
      }.toMap
    }

    val pairwiseStats = targetAttrPairsToComputeStats.map { case attrPair @ (rvX, rvY) =>
      // The conditional entropy is 0 for strongly correlated attributes and 1 for completely independent
      // attributes. We reverse this to reflect the correlation.
      val domainSize = domainStatMap(rvX)
      attrPair -> (1.0 - ((hXYs(Set(rvX, rvY)) - hYs(rvY)) / scala.math.log10(domainSize)))
    }
    pairwiseStats.groupBy { case ((attrToRepair, _), _) =>
      attrToRepair
    }.map { case (k, v) =>
      k -> v.map { case ((_, attr), v) =>
        (attr, v)
      }.sortBy(_._2).reverse
    }
  }

  private[python] def filterCorrAttrs(
      pairwiseStatMap: Map[String, Seq[(String, Double)]],
      maxAttrsToComputeDomains: Int,
      minCorrThres: Double): Map[String, Seq[(String, Double)]] = {
    assert(pairwiseStatMap.nonEmpty)
    assert(maxAttrsToComputeDomains > 0)
    assert(minCorrThres >= 0.0)

    logBasedOnLevel({
      val pairStats = pairwiseStatMap.map { case (k, v) =>
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

    pairwiseStatMap.map { case (k, v) =>
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
  }

  def computeDomainInErrorCells(
      discretizedInputView: String,
      errCellView: String,
      rowId: String,
      continuousAttrList: String,
      targetAttrList: String,
      discretizedAttrList: String,
      rowCount: Long,
      maxAttrsToComputeDomains: Int,
      statSampleRatio: Double,
      statThreshold: Double,
      minCorrThres: Double,
      domain_threshold_alpha: Double,
      domain_threshold_beta: Double): String = {
    assert(0 < rowCount, "rowCount should be greater than 0.")
    assert(0 < maxAttrsToComputeDomains, "maxAttrsToComputeDomains should be greater than 0.")
    assert(0.0 <= minCorrThres && minCorrThres < 1.0, "minCorrThres should be in [0.0, 1.0).")
    assert(0.0 <= domain_threshold_alpha && domain_threshold_alpha <= 1.0,
      "domain_threashold_alpha should be in [0.0, 1.0].")
    assert(0.0 <= domain_threshold_beta && domain_threshold_beta <= 1.0,
      "domain_threashold_beta should be in [0.0, 1.0].")

    logBasedOnLevel(s"computeDomainInErrorCells called with: " +
      s"discretizedInputView=$discretizedInputView errCellView=$errCellView rowId=$rowId " +
      s"continousAttrList=${if (continuousAttrList.nonEmpty) continuousAttrList else "<none>"} " +
      s"targetAttrList=$targetAttrList discretizedAttrList=$discretizedAttrList " +
      s"rowCount=$rowCount maxAttrsToComputeDomains=$maxAttrsToComputeDomains " +
      s"statSampleRatio=$statSampleRatio statThreshold=$statThreshold minCorrThres=$minCorrThres " +
      s"domain_threshold=alpha:$domain_threshold_alpha,beta=$domain_threshold_beta")

    val discretizedAttrs = SparkUtils.stringToSeq(discretizedAttrList)
    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)

    assert(checkSchema(errCellView, "attribute STRING, current_value STRING", rowId, strict = true))
    assert(discretizedAttrs.nonEmpty && attrsToRepair.nonEmpty)

    val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
      discretizedAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
    }

    val attrsToComputeFreqStats = discretizedAttrs.map(a => Seq(a)) ++
      attrPairsToRepair.map { case (a1, a2) => Seq(a1, a2) }
    val attrStatDf = computeFreqStats(
      discretizedInputView, attrsToComputeFreqStats, statSampleRatio, statThreshold)

    withTempView(attrStatDf, cache = true) { attrStatView =>
      val domainStatMap = computeAndGetTableStats(discretizedInputView).mapValues(_.distinctCount)
      val pairwiseStatMap = computePairwiseStats(
        discretizedInputView, rowCount, attrStatView, discretizedAttrs,
        attrPairsToRepair, domainStatMap)
      val corrAttrMap = filterCorrAttrs(pairwiseStatMap, maxAttrsToComputeDomains, minCorrThres)
      val domainInitValue = s"CAST(NULL AS ARRAY<STRUCT<n: STRING, cnt: DOUBLE>>)"
      val repairCellDf = spark.table(errCellView).where(
        s"attribute IN (${attrsToRepair.map(a => s"'$a'").mkString(", ")})")
      val cellDomainDf = if (domain_threshold_beta >= 1.0) {
        // The case where we don't need to compute error domains
        withTempView(repairCellDf) { repairCellView =>
          spark.sql(
            s"""
               |SELECT
               |  l.$rowId, r.attribute, r.current_value, $domainInitValue domain
               |FROM
               |  $discretizedInputView l, $repairCellView r
               |WHERE
               |  l.$rowId = r.$rowId
             """.stripMargin)
        }
      } else {
        // Needs to keep the correlated attributes for selecting their domains
        val corrAttrSet = corrAttrMap.flatMap(_._2.map(_._1)).toSet
        val corrAttrs = if (corrAttrSet.nonEmpty) {
          corrAttrSet.mkString(", ", ", ", "")
        } else {
          ""
        }

        val rvDf = withTempView(repairCellDf) { repairCellView =>
          spark.sql(
            s"""
               |SELECT
               |  l.$rowId, r.attribute, r.current_value $corrAttrs
               |FROM
               |  $discretizedInputView l, $repairCellView r
               |WHERE
               |  l.$rowId = r.$rowId
             """.stripMargin)
        }

        withTempView(rvDf) { rvView =>
          val continousAttrs = SparkUtils.stringToSeq(continuousAttrList).toSet
          corrAttrMap.map { case (attribute, corrAttrsWithScores) =>
            // Adds an empty domain for initial state
            val initDomainDf = spark.sql(
              s"""
                 |SELECT $rowId, attribute, current_value, $domainInitValue domain $corrAttrs
                 |FROM $rvView
                 |WHERE attribute = '$attribute'
               """.stripMargin)

            val domainDf = if (!continousAttrs.contains(attribute) && corrAttrsWithScores.nonEmpty) {
              val corrAttrs = corrAttrsWithScores.map(_._1)
              logBasedOnLevel(s"Computing '$attribute' domain from ${corrAttrs.size} correlated " +
                s"attributes (${corrAttrs.mkString(",")})...")

              corrAttrs.foldLeft(initDomainDf) { case (df, attr) =>
                withTempView(df) { domainSpaceView =>
                  val tau = {
                    // `tau` becomes a threshold on co-occurrence frequency
                    val productSpaceSize = domainStatMap(attr) * domainStatMap(attribute)
                    (domain_threshold_alpha * (rowCount / productSpaceSize)).toLong
                  }
                  spark.sql(
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
              spark.sql(
                s"""
                   |SELECT
                   |  $rowId, attribute, current_value, domain_value, SUM(score) score
                   |FROM (
                   |  SELECT
                   |    $rowId,
                   |    attribute,
                   |    current_value,
                   |    domain_value_with_freq.n domain_value,
                   |    exp(ln(cnt / $rowCount) + ln(domain_value_with_freq.cnt / cnt)) score
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
                   |    WHERE ${whereCaluseToFilterStat(attribute, discretizedAttrs)}
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
              spark.sql(
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
      Seq("cell_domain" -> cellDomainView,
        "pairwise_attr_stats" -> pairwiseStatMap.mapValues(seqToJson)
      ).asJson
    }
  }
}
