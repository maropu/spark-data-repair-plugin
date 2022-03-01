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

import scala.util.control.NonFatal

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.python.RegexStructureRepair
import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql._
import org.apache.spark.util.RepairUtils._
import org.apache.spark.util.{Utils => SparkUtils}

/** A Python API entry point for data cleaning. */
object RepairApi extends RepairBase {

  def checkInputTable(dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"checkInputTable called with: dbName=$dbName " +
      s"tableName=$tableName rowId=$rowId")

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
    val distinctCount = inputDf.selectExpr(s"`$rowId`").distinct().count()
    if (distinctCount != rowCount) {
      throw AnalysisException(s"Uniqueness does not hold in column '$rowId' " +
        s"of table '$qualifiedName' (# of distinct '$rowId': $distinctCount, # of rows: $rowCount)")
    }

    val continousAttrs = inputDf.schema
      .filter(f => f.name != rowId && continousTypes.contains(f.dataType))
      .map(_.name).mkString(",")

    Seq("input_table" -> qualifiedName, "continous_attrs" -> continousAttrs).asJson
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
    val cellExprs = targetAttrs.map { a => s"CAST(r.`$a` AS STRING) `$a`" }
    val df = spark.sql(
      s"""
         |SELECT
         |  l.`$rowId`,
         |  l.attribute,
         |  extractField(struct(${cellExprs.mkString(", ")}), l.attribute) current_value
         |FROM
         |  $errCellView l, $inputView r
         |WHERE
         |  l.`$rowId` = r.`$rowId`
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
    Seq("domain_stats" -> statMap.mapValues(_.distinctCount.toString)).asJson
  }

  private def discretizeTable(
      inputView: String,
      rowId: String,
      targetAttrs: Seq[String],
      statMap: Map[String, ColumnStat],
      discreteThreshold: Int): DataFrame = {
    val inputDf = spark.table(inputView)
    val attrTypeMap = inputDf.schema.map { f => f.name -> f.dataType }.toMap
    val discretizedExprs = targetAttrs.flatMap { attr =>
      (statMap(attr), attrTypeMap(attr)) match {
        case (ColumnStat(_, min, max), tpe) if continousTypes.contains(tpe) =>
          logBasedOnLevel(s"'$attr' regraded as a continuous attribute (min=${min.get}, " +
            s"max=${max.get}), so discretized into [0, $discreteThreshold)")
          Some(s"int((`$attr` - ${min.get}) / (${max.get} - ${min.get}) * $discreteThreshold) `$attr`")
        case (ColumnStat(distinctCount, _, _), _)
          if 1 < distinctCount && distinctCount <= discreteThreshold =>
          Some(s"`$attr`")
        case (ColumnStat(distinctCount, _, _), _) =>
          logWarning(s"'$attr' dropped because of its unsuitable domain (size=$distinctCount)")
          None
      }
    }
    inputDf.selectExpr(s"`$rowId`" +: discretizedExprs: _*)
  }

  def convertToDiscretizedTable(
      qualifiedName: String,
      rowId: String,
      discreteThreshold: Int): String = {
    logBasedOnLevel(s"convertToDiscretizedTable called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId discreteThreshold=$discreteThreshold")

    assert(rowId.nonEmpty, s"$rowId should be a non-empty string.")
    assert(2 <= discreteThreshold && discreteThreshold < 65536, "discreteThreshold should be in [2, 65536).")

    val targetAttrs = spark.table(qualifiedName).columns.filter(_ != rowId).toSeq
    val statMap = computeAndGetTableStats(qualifiedName).filterKeys(_ != rowId)
    val discreteDf = discretizeTable(qualifiedName, rowId, targetAttrs, statMap, discreteThreshold)
    val distinctStats = statMap.mapValues(_.distinctCount.toString)
    val discretizedView = createTempView(discreteDf, "discretized_table", cache = true)
    Seq("discretized_table" -> discretizedView,
      "domain_stats" -> distinctStats
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
         |SELECT `$rowId`, collect_set(attribute) AS errors
         |FROM $errCellView
         |GROUP BY `$rowId`
       """.stripMargin)

    val repairBase = withTempView(errAttrDf, "error_attrs") { errAttrView =>
      val cleanAttrs = spark.table(discretizedInputView).columns.map {
        case attr if attr == rowId =>
          s"$discretizedInputView.`$rowId`"
        case attr if attrsToRepair.contains(attr) =>
          s"IF(array_contains(errors, '$attr'), NULL, `$attr`) AS `$attr`"
        case cleanAttr =>
          s"`$cleanAttr`"
      }
      spark.sql(
        s"""
           |SELECT ${cleanAttrs.mkString(", ")}
           |FROM $discretizedInputView
           |LEFT OUTER JOIN $errAttrView
           |ON $discretizedInputView.`$rowId` = $errAttrView.`$rowId`
         """.stripMargin)
    }
    val repairBaseView = createTempView(repairBase, "repair_base_cells")
    Seq("repair_base_cells" -> repairBaseView).asJson
  }

  def computeFunctionalDeps(
      inputView: String,
      constraintFilePath: String,
      constraints: String,
      targetAttrList: String): String = {
    logBasedOnLevel(s"computeFunctionalDeps called with: discretizedInputView=$inputView " +
      s"constraintFilePath=$constraintFilePath targetAttrList=$targetAttrList")
    val targetAttrs = SparkUtils.stringToSeq(targetAttrList)
    DepGraph.computeFunctionalDeps(inputView, constraintFilePath, constraints, targetAttrs)
  }

  def computeFunctionalDepMap(inputView: String, x: String, y: String): String = {
    logBasedOnLevel(s"computeFunctionalDepMap called with: inputView=$inputView x=$y x=$y")
    DepGraph.computeFunctionalDepMap(inputView, x, y)
  }

  private[python] def computeFreqStats(
      inputView: String,
      targetAttrSets: Seq[Seq[String]],
      attrFreqRatioThreshold: Double): DataFrame = {
    assert(targetAttrSets.nonEmpty)
    assert(0.0 <= attrFreqRatioThreshold && attrFreqRatioThreshold <= 1.0)

    val targetAttrs = targetAttrSets.flatten.distinct
    val distinctTargetAttrSet = targetAttrSets.map(_.toSet).distinct.map(_.toSeq)
    // TODO: Needs to support more larger cases for grouping attributes
    if (targetAttrs.length > 64) {
      throw AnalysisException("Cannot handle the target attributes whose length is more than 64, " +
        s"""but got: ${targetAttrs.mkString(",")}""")
    }

    val groupingSetSeq = distinctTargetAttrSet.map {
      case Seq(a) => s"(`$a`)"
      case Seq(a1, a2) => s"(`$a1`,`$a2`)"
      case attrs =>
        throw new IllegalStateException(
          s"Cannot handle more than two entries: ${attrs.mkString(",")}")
    }

    withTempView(spark.table(inputView), "input_to_compute_freq_stats") { inputView =>
      val filterClauseOption = if (attrFreqRatioThreshold > 0.0) {
        val rowCount = spark.table(inputView).count()
        val cond = s"HAVING cnt > ${(rowCount * attrFreqRatioThreshold).toInt}"
        logBasedOnLevel(s"Attributes stats filter enabled: $cond")
        cond
      } else {
        ""
      }
      spark.sql(
        s"""
           |SELECT ${targetAttrs.map(a => s"`$a`").mkString(", ")}, COUNT(1) cnt
           |FROM $inputView
           |GROUP BY GROUPING SETS (
           |  ${groupingSetSeq.mkString(", ")}
           |)
           |$filterClauseOption
         """.stripMargin)
    }
  }

  private def whereCaluseToFilterStat(a: String, attrs: Seq[String]): String = {
    s"`$a` IS NOT NULL AND ${attrs.filter(_ != a).map(a => s"`$a` IS NULL").mkString(" AND ")}"
  }

  private def log2(v: Double): Double = {
    Math.log(v) / Math.log(2.0)
  }

  private[python] def computePairwiseStats(
       rowCount: Long,
       freqStatView: String,
       targetAttrPairsToComputeStats: Seq[(String, String)],
       domainStatMap: Map[String, Long]): Map[String, Seq[(String, Double)]] = {
    if (targetAttrPairsToComputeStats.isEmpty) {
      Map.empty
    } else {
      val freqStatAttrs = spark.table(freqStatView).columns.filter(_ != "cnt")
      val targetAttrs = targetAttrPairsToComputeStats.flatMap(p => Seq(p._1, p._2)).distinct

      assert(rowCount > 0)
      assert((freqStatAttrs.toSet & targetAttrs.toSet) == targetAttrs.toSet)
      assert(targetAttrs.forall(domainStatMap.contains))

      // Computes the conditional entropy: H(x|y) = H(x,y) - H(y).
      // If H(x|y) = 0, then y determines x, i.e., y -> x.
      val hXYs = withJobDescription("compute conditional entropy H(x,y)") {
        val attrPairSets = targetAttrPairsToComputeStats.map(p => Set(p._1, p._2)).distinct
        attrPairSets.map { attrPair =>
          attrPair -> {
            val Seq(x, y) = attrPair.toSeq
            val corrTerm = {
              val df = spark.sql(
                s"""
                   |SELECT COUNT(1), COALESCE(SUM(cnt), 0)
                   |FROM $freqStatView
                   |WHERE `$x` IS NOT NULL AND
                   |  `$y` IS NOT NULL
                 """.stripMargin)
              val (domainSize, totalCount) = df.take(1).map {
                case Row(cnt: Long, sum: Long) => (cnt, sum)
              }.head

              if (rowCount > totalCount) {
                val ubDomainSize = Math.max(domainStatMap(x) * domainStatMap(y) - domainSize, 1)
                val avgCnt = Math.max((rowCount - totalCount + 0.0) / ubDomainSize, 1.0)
                -ubDomainSize * (avgCnt / rowCount) * log2(avgCnt / rowCount)
              } else {
                0.0
              }
            }

            val hXY = getRandomString(prefix="hXY")
            val df = spark.sql(
              s"""
                 |SELECT -COALESCE(SUM($hXY), 0.0) $hXY
                 |FROM (
                 |  SELECT `$x` X, `$y` Y, (cnt / $rowCount) * log2(cnt / $rowCount) $hXY
                 |  FROM $freqStatView
                 |  WHERE `$x` IS NOT NULL AND
                 |    `$y` IS NOT NULL
                 |)
               """.stripMargin)

            df.take(1).head.getDouble(0) + corrTerm
          }
        }.toMap
      }

      val hYs = withJobDescription("compute entropy H(y)") {
        targetAttrs.map { attr =>
          attr -> {
            val corrTerm = {
              val df = spark.sql(
                s"""
                   |SELECT COUNT(1), COALESCE(SUM(cnt), 0)
                   |FROM $freqStatView
                   |WHERE ${whereCaluseToFilterStat(attr, freqStatAttrs)}
                 """.stripMargin)
              val (domainSize, totalCount) = df.take(1).map {
                case Row(cnt: Long, sum: Long) => (cnt, sum)
              }.head

              if (rowCount > totalCount) {
                val ubDomainSize = Math.max(domainStatMap(attr) - domainSize, 1)
                val avgCnt = Math.max((rowCount - totalCount + 0.0) / ubDomainSize, 1.0)
                -ubDomainSize * (avgCnt / rowCount) * log2(avgCnt / rowCount)
              } else {
                0.0
              }
            }

            val hY = getRandomString(prefix="hY")
            val df = spark.sql(
              s"""
                 |SELECT -COALESCE(SUM($hY), 0.0) $hY
                 |FROM (
                 |  SELECT `$attr` Y, (cnt / $rowCount) * log2(cnt / $rowCount) $hY
                 |  FROM $freqStatView
                 |  WHERE ${whereCaluseToFilterStat(attr, freqStatAttrs)}
                 |)
               """.stripMargin)

            df.take(1).head.getDouble(0) + corrTerm
          }
        }.toMap
      }

      val pairwiseStats = targetAttrPairsToComputeStats.map { case attrPair @ (rvX, rvY) =>
        attrPair -> (hXYs(Set(rvX, rvY)) - hYs(rvY))
      }
      pairwiseStats.groupBy { case ((attrToRepair, _), _) =>
        attrToRepair
      }.map { case (k, v) =>
        k -> v.map { case ((_, attr), v) =>
          (attr, v)
        }.sortBy(_._2)
      }
    }
  }

  def computeAttrStats(
      discretizedInputView: String,
      rowId: String,
      targetAttrList: String,
      domainStatMapAsJson: String,
      attrFreqRatioThreshold: Double,
      pairwiseFreqRatioThreshold: Double,
      maxAttrsToComputePairwiseStats: Int): String = {
    logBasedOnLevel(s"computeAttrStats called with: " +
      s"discretizedInputView=$discretizedInputView rowId=$rowId " +
      s"targetAttrList=$targetAttrList attrFreqRatioThreshold=$attrFreqRatioThreshold " +
      s"pairwiseFreqRatioThreshold=$pairwiseFreqRatioThreshold " +
      s"maxAttrsToComputePairwiseStats=$maxAttrsToComputePairwiseStats")

    assert(0.0 <= attrFreqRatioThreshold && attrFreqRatioThreshold <= 1.0,
      "attrFreqRatioThreshold should be in [0.0, 1.0].")
    assert(0.0 <= pairwiseFreqRatioThreshold && pairwiseFreqRatioThreshold <= 1.0,
      "pairwiseFreqRatioThreshold should be in [0.0, 1.0].")
    assert(0 < maxAttrsToComputePairwiseStats,
      "maxAttrsToComputePairwiseStats should be greater than 0.")

    val discretizedAttrs = spark.table(discretizedInputView).columns.filter(_ != rowId)
    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)
    assert(attrsToRepair.nonEmpty)

    val domainStatMap = {
      val jsonObj = parse(domainStatMapAsJson)
      val data = jsonObj.asInstanceOf[JObject].values.asInstanceOf[Map[String, scala.math.BigInt]]
      val mapData = data.mapValues(_.toLong)
      assert(attrsToRepair.forall(mapData.contains))
      mapData
    }

    // Filters the attribute pairs that might have high correlation between each other
    val candidateAttrPairs = attrsToRepair.flatMap { attrToRepair =>
      val candidates = discretizedAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      if (candidates.length > maxAttrsToComputePairwiseStats) {
        val candidatesWithScores = candidates.map { case attrPair @ (x, y) =>
          val coRowCount = spark.table(discretizedInputView)
            .selectExpr(s"approx_count_distinct(struct(`$x`, `$y`)) cnt")
            .take(1).head.getLong(0)
          val coRatio = (coRowCount + 0.0) / (domainStatMap(x) * domainStatMap(y))
          (coRatio, attrPair)
        }
        candidatesWithScores
          .filter(_._1 < pairwiseFreqRatioThreshold)
          .sortBy(_._1)
          .take(maxAttrsToComputePairwiseStats)
          .map(_._2)
      } else {
        candidates
      }
    }

    val freqAttrStatView = {
      val targetAttrSets = discretizedAttrs.map(a => Seq(a)) ++
        candidateAttrPairs.map { p => Seq(p._1, p._2) }
      createTempView(
        computeFreqStats(discretizedInputView, targetAttrSets, attrFreqRatioThreshold),
        "attr_freq_stats",
        cache = true)
    }

    val pairwiseStatMap = {
      val rowCount = spark.table(discretizedInputView).count()
      val statMap = computePairwiseStats(
        rowCount, freqAttrStatView, candidateAttrPairs, domainStatMap)
      val missingKeys = attrsToRepair.toSet.diff(statMap.keySet)
      if (missingKeys.nonEmpty) {
        statMap ++ missingKeys.map { k => (k, Seq.empty[(String, Double)]) }
      } else {
        statMap
      }
    }

    assert(pairwiseStatMap.keySet == attrsToRepair.toSet)

    Seq(
      "attr_freq_stats" -> freqAttrStatView,
      "pairwise_attr_corr_stats" -> pairwiseStatMap.mapValues(seqToJson)
    ).asJson
  }

  def computeDomainInErrorCells(
      discretizedInputView: String,
      errCellView: String,
      rowId: String,
      continuousAttrList: String,
      targetAttrList: String,
      freqAttrStatView: String,
      pairwiseStatMapAsJson: String,
      domainStatMapAsJson: String,
      maxAttrsToComputeDomains: Int,
      domainThresholdAlpha: Double,
      domainThresholdBeta: Double): DataFrame = {
    logBasedOnLevel(s"computeDomainInErrorCells called with: " +
      s"discretizedInputView=$discretizedInputView errCellView=$errCellView rowId=$rowId " +
      s"continousAttrList=${if (continuousAttrList.nonEmpty) continuousAttrList else "<none>"} " +
      s"targetAttrList=$targetAttrList freqAttrStatView=$freqAttrStatView " +
      s"maxAttrsToComputeDomains=$maxAttrsToComputeDomains " +
      s"domain_threshold=alpha:$domainThresholdAlpha,beta=$domainThresholdBeta")

    assert(spark.table(discretizedInputView).columns.length > 1)
    assert(checkSchema(errCellView, "attribute STRING, current_value STRING", rowId, strict = true))
    assert(0 < maxAttrsToComputeDomains, "maxAttrsToComputeDomains should be greater than 0.")
    assert(0.0 <= domainThresholdAlpha && domainThresholdAlpha < 1.0,
      "domainThresholdAlpha should be in [0.0, 1.0).")
    assert(0.0 <= domainThresholdBeta && domainThresholdBeta < 1.0,
      "domainThresholdBeta should be in [0.0, 1.0).")
    assert(domainThresholdAlpha < domainThresholdBeta,
      "domainThresholdAlpha should be greater than domainThresholdBeta.")

    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)
    assert(attrsToRepair.nonEmpty)

    val domainStatMap = {
      val jsonObj = parse(domainStatMapAsJson)
      val data = jsonObj.asInstanceOf[JObject].values.asInstanceOf[Map[String, scala.math.BigInt]]
      val mapData = data.mapValues(_.toLong)
      assert(attrsToRepair.forall(mapData.contains))
      mapData
    }

    val pairwiseStatMap = {
      val jsonObj = parse(pairwiseStatMapAsJson)
      val data = jsonObj.asInstanceOf[JObject].values.asInstanceOf[Map[String, Seq[Seq[String]]]]
      val mapData = data.mapValues(_.map { case Seq(attr, sv) => (attr, sv.toDouble) })
      assert(attrsToRepair.forall(mapData.contains))
      mapData
    }

    withJobDescription("compute domain values with posteriori probability") {
      val corrAttrMap = pairwiseStatMap.map { case (k, attrs) => (k, attrs.take(maxAttrsToComputeDomains)) }
      val domainInitValue = s"CAST(NULL AS ARRAY<STRUCT<n: STRING, cnt: DOUBLE>>)"
      val repairCellDf = spark.table(errCellView).where(
        s"attribute IN (${attrsToRepair.map(a => s"'$a'").mkString(", ")})")

      // Needs to keep the correlated attributes for selecting their domains
      val corrAttrSet = corrAttrMap.flatMap(_._2.map(_._1)).toSet
      val corrAttrs = if (corrAttrSet.nonEmpty) {
        corrAttrSet.map(c => s"`$c`").mkString(", ", ", ", "")
      } else {
        ""
      }

      val rvDf = withTempView(repairCellDf, "repair_cells") { repairCellView =>
        spark.sql(
          s"""
             |SELECT
             |  l.`$rowId`, r.attribute, r.current_value $corrAttrs
             |FROM
             |  $discretizedInputView l, $repairCellView r
             |WHERE
             |  l.`$rowId` = r.`$rowId`
           """.stripMargin)
      }

      withTempView(rvDf, "rv", cache = true) { rvView =>
        val rowCount = spark.table(discretizedInputView).count()
        val continousAttrs = SparkUtils.stringToSeq(continuousAttrList).toSet
        corrAttrMap.map { case (attribute, corrAttrsWithScores) =>
          // Adds an empty domain for initial state
          val initDomainDf = spark.sql(
            s"""
               |SELECT `$rowId`, attribute, current_value, $domainInitValue domain $corrAttrs
               |FROM $rvView
               |WHERE attribute = '$attribute'
             """.stripMargin)

          val domainDf = if (!continousAttrs.contains(attribute) && corrAttrsWithScores.nonEmpty) {
            val corrAttrs = corrAttrsWithScores.map(_._1)
            logBasedOnLevel(s"Computing '$attribute' domain from ${corrAttrs.size} correlated " +
              s"attributes (${corrAttrs.mkString(",")})...")

            corrAttrs.foldLeft(initDomainDf) { case (df, attr) =>
              withTempView(df, "domain_space") { domainSpaceView =>
                val tau = {
                  // `tau` becomes a threshold on co-occurrence frequency
                  val productSpaceSize = domainStatMap(attr) * domainStatMap(attribute)
                  (domainThresholdAlpha * (rowCount / productSpaceSize)).toLong
                }
                spark.sql(
                  s"""
                     |SELECT
                     |  `$rowId`,
                     |  attribute,
                     |  current_value,
                     |  IF(ISNOTNULL(l.domain), CONCAT(l.domain, r.d), r.d) domain,
                     |  ${corrAttrSet.map(a => s"l.`$a`").mkString(",")}
                     |FROM
                     |  $domainSpaceView l
                     |LEFT OUTER JOIN (
                     |  SELECT `$attr`, collect_set(named_struct('n', `$attribute`, 'cnt', array_max(array(double(cnt) - 1.0, 0.1)))) d
                     |  FROM (
                     |    SELECT *
                     |    FROM $freqAttrStatView
                     |    WHERE `$attribute` IS NOT NULL AND
                     |      `$attr` IS NOT NULL AND
                     |      cnt > $tau
                     |  )
                     |  GROUP BY
                     |    `$attr`
                     |) r
                     |ON
                     |  l.`$attr` = r.`$attr`
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
          val score = getRandomString(prefix="score")
          val domainWithScoreDf = withTempView(domainDf, "domain") { domainView =>
            val discretizedAttrs = spark.table(discretizedInputView).columns.filter(_ != rowId)
            spark.sql(
              s"""
                 |SELECT
                 |  `$rowId`, attribute, current_value, domain_value, SUM($score) $score
                 |FROM (
                 |  SELECT
                 |    `$rowId`,
                 |    attribute,
                 |    current_value,
                 |    domain_value_with_freq.n domain_value,
                 |    exp(ln(cnt / $rowCount) + ln(domain_value_with_freq.cnt / cnt)) $score
                 |  FROM (
                 |    SELECT
                 |      `$rowId`,
                 |      attribute,
                 |      current_value,
                 |      explode_outer(domain) domain_value_with_freq
                 |    FROM
                 |      $domainView
                 |  ) d LEFT OUTER JOIN (
                 |    SELECT `$attribute`, MAX(cnt) cnt
                 |    FROM $freqAttrStatView
                 |    WHERE ${whereCaluseToFilterStat(attribute, discretizedAttrs)}
                 |    GROUP BY `$attribute`
                 |  ) s
                 |  ON
                 |    d.domain_value_with_freq.n = s.`$attribute`
                 |)
                 |GROUP BY
                 |  `$rowId`, attribute, current_value, domain_value
               """.stripMargin)
          }

          withTempView(domainWithScoreDf, "domain_with_scores", cache = true) { domainWithScoreView =>
            val denom = getRandomString(prefix="denom")
            spark.sql(
              s"""
                 |SELECT
                 |  l.`$rowId`,
                 |  l.attribute,
                 |  current_value,
                 |  filter(collect_set(named_struct('n', domain_value, 'prob', $score / $denom)), x -> x.prob > $domainThresholdBeta) domain
                 |FROM
                 |  $domainWithScoreView l, (
                 |    SELECT
                 |      `$rowId`, attribute, SUM($score) $denom
                 |    FROM
                 |      $domainWithScoreView
                 |    GROUP BY
                 |      `$rowId`, attribute
                 |  ) r
                 |WHERE
                 |  l.`$rowId` = r.`$rowId` AND l.attribute = r.attribute
                 |GROUP BY
                 |  l.`$rowId`, l.attribute, current_value
               """.stripMargin)
          }
        }
      }.reduce(_.union(_))
    }
  }

  def repairByRegularExpression(
      regex: String,
      targetAttr: String,
      errCellView: String,
      rowId: String): DataFrame = {
    logBasedOnLevel(s"repairByRegularExpression called with: " +
      s"regex=$regex targetAttr=$targetAttr errCellView=$errCellView rowId=$rowId")

    assert(regex.nonEmpty)
    assert(targetAttr.nonEmpty)
    assert(checkSchema(errCellView, "attribute STRING, current_value STRING", rowId, strict = true))
    assert(rowId.nonEmpty)

    import functions._
    val inputDf = spark.table(errCellView)
    try {
      val repair = RegexStructureRepair(regex)
      val repairUdf = udf((s: String) => repair(s).orNull)
      inputDf.select(col(rowId), col("attribute"), col("current_value"),
        when(expr(s"attribute = '$targetAttr'"), repairUdf(col("current_value")))
          .otherwise(expr("null")).as("repaired"))
    } catch {
      case NonFatal(e) =>
        logWarning(s"Repairing using regex '$regex' (attr='$targetAttr') " +
          s"failed because: ${e.getMessage}")
        inputDf.withColumn("repaired", expr("string(null)"))
    }
  }
}
