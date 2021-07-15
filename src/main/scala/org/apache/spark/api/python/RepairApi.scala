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

import java.net.URI

import scala.collection.mutable
import scala.io.Source

import org.apache.spark.python.DenialConstraints
import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql._
import org.apache.spark.sql.types.StringType
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
    val rowCnt = inputDf.count()
    val distinctCnt = inputDf.selectExpr(rowId).distinct().count()
    if (distinctCnt != rowCnt) {
      throw AnalysisException(s"Uniqueness does not hold in column '$rowId' " +
        s"of table '$qualifiedName' (# of distinct '$rowId': $distinctCnt, # of rows: $rowCnt)")
    }

    val continousAttrs = inputDf.schema.filter(f => continousTypes.contains(f.dataType))
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
    logBasedOnLevel(s"extractCurrentValues called with: inputView=$inputView " +
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

  def computeDomainSizes(discreteAttrView: String): String = {
    logBasedOnLevel(s"computeDomainSizes called with: discreteAttrView=$discreteAttrView")

    val statMap = computeAndGetTableStats(discreteAttrView)
    Seq("distinct_stats" -> statMap.mapValues(_.distinctCount.toString)).asJson
  }

  def computeFunctionalDeps(inputView: String, constraintFilePath: String): String = {
    logBasedOnLevel(s"computeFunctionalDep called with: discreteAttrView=$inputView " +
      s"constraintFilePath=$constraintFilePath")

    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName("", inputView)

    var file: Source = null
    val constraints = try {
      file = Source.fromFile(new URI(constraintFilePath).getPath)
      file.getLines()
      DenialConstraints.parseAndVerifyConstraints(file.getLines(), qualifiedName, inputDf.columns)
    } finally {
      if (file != null) {
        file.close()
      }
    }

    // TODO: Reuse the previous computation result
    val domainSizes = {
      // TODO: `StringType` only supported now
      val supported = inputDf.schema.filter(_.dataType == StringType).map(_.name).toSet
      computeAndGetTableStats(inputView).mapValues(_.distinctCount)
        .filter(kv => supported.contains(kv._1))
    }

    val fdMap = mutable.Map[String, mutable.Set[String]]()

    def hasNoCyclic(ref1: String, ref2: String): Boolean = {
      !fdMap.get(ref1).exists(_.contains(ref2)) && !fdMap.get(ref2).exists(_.contains(ref1))
    }

    constraints.predicates.filter { preds =>
      // Filters predicate candidates that might mean functional deps
      preds.length == 2 && preds.flatMap(_.references).distinct.length == 2
    }.foreach {
      case Seq(p1, p2) if Set(p1.sign, p2.sign) == Set("EQ", "IQ") &&
          p1.references.length == 1 && p2.references.length == 1 =>
        val ref1 = p1.references.head
        val ref2 = p2.references.head
        (domainSizes.get(ref1), domainSizes.get(ref2)) match {
          case (Some(ds1), Some(ds2)) if ds1 < ds2 =>
            fdMap.getOrElseUpdate(ref1, mutable.Set[String]()) += ref2
          case (Some(ds1), Some(ds2)) if ds1 > ds2 =>
            fdMap.getOrElseUpdate(ref2, mutable.Set[String]()) += ref1
          case (Some(_), Some(_)) if hasNoCyclic(ref1, ref2) =>
            fdMap(ref1) = mutable.Set(ref2)
          case _ =>
        }
      case _ =>
    }

    // TODO: We need a smarter way to convert Scala data to a json string
    fdMap.map { case (k, values) =>
      s""""$k": [${values.toSeq.sorted.map { v => s""""$v"""" }.mkString(",")}]"""
    }.mkString("{", ",", "}")
  }

  def computeFunctionDepMap(inputView: String, X: String, Y: String): String = {
    val df = spark.sql(
      s"""
         |SELECT CAST($X AS STRING) x, CAST(y[0] AS STRING) y FROM (
         |  SELECT $X, collect_set($Y) y
         |  FROM $inputView
         |  GROUP BY $X
         |  HAVING size(y) = 1
         |)
       """.stripMargin)

    // TODO: We need a smarter way to convert Scala data to a json string
    df.collect.map { case Row(x: String, y: String) =>
      s""""$x": "$y""""
    }.mkString("{", ",", "}")
  }

  private def discretizeTable(
      inputView: String,
      discreteThres: Int,
      whitelist: Set[String] = Set.empty): (DataFrame, Map[String, ColumnStat]) = {
    assert(2 <= discreteThres && discreteThres < 65536, "discreteThres should be in [2, 65536).")
    val statMap = computeAndGetTableStats(inputView)
    val inputDf = spark.table(inputView)
    val attrTypeMap = inputDf.schema.map { f => f.name -> f.dataType }.toMap
    val discreteExprs = inputDf.columns.flatMap { attr =>
      (statMap(attr), attrTypeMap(attr)) match {
        case (ColumnStat(_, min, max), tpe) if continousTypes.contains(tpe) =>
          logBasedOnLevel(s"'$attr' regraded as a continuous attribute (min=${min.get}, " +
            s"max=${max.get}), so discretized into [0, $discreteThres)")
          Some(s"int(($attr - ${min.get}) / (${max.get} - ${min.get}) * $discreteThres) $attr")
        case (ColumnStat(distinctCnt, _, _), _)
          if whitelist.contains(attr) || (1 < distinctCnt && distinctCnt < discreteThres) =>
          Some(attr)
        case (ColumnStat(distinctCnt, _, _), _) =>
          logWarning(s"'$attr' dropped because of its unsuitable domain (size=$distinctCnt)")
          None
      }
    }
    val discreteDf = inputDf.selectExpr(discreteExprs: _*)
    (discreteDf, statMap)
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
    val discreteFeaturesView = createAndCacheTempView(discreteDf, "discretized_table")
    Seq("discretized_table" -> discreteFeaturesView,
      "distinct_stats" -> distinctStats
    ).asJson
  }

  def convertToHistogram(inputView: String, discreteThres: Int): String = {
    logBasedOnLevel(s"convertToHistogram called with: inputView=$inputView " +
      s"discreteThres=$discreteThres")

    val (discreteDf, _) = discretizeTable(inputView, discreteThres)

    val histogramDf = {
      withTempView(discreteDf, cache = true) { discreteView =>
        val sqls = discreteDf.columns.map { attr =>
          s"""
             |SELECT '$attr' attribute, collect_list(b) histogram
             |FROM (
             |  SELECT named_struct('value', $attr, 'cnt', COUNT(1)) b
             |  FROM $discreteView
             |  GROUP BY $attr
             |)
           """.stripMargin
        }
        spark.sql(sqls.mkString(" UNION ALL "))
      }
    }
    val histgramView = createAndCacheTempView(histogramDf, "histogram")
    Seq("histogram" -> histgramView).asJson
  }

  def convertErrorCellsToNull(
      discreteAttrView: String,
      errCellView: String,
      rowId: String,
      targetAttrList: String): String = {
    logBasedOnLevel(s"convertErrorCellsToNull called with: discreteAttrView=$discreteAttrView " +
      s"errCellView=$errCellView rowId=$rowId targetAttrList=$targetAttrList")

    // `errCellView` schema must have `$rowId`, `attribute`, and `current_value` columns
    assert(checkSchema(errCellView, "attribute STRING, current_value STRING", rowId, strict = true))

    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)
    assert(attrsToRepair.nonEmpty)

    val errAttrDf = spark.sql(
      s"""
         |SELECT $rowId, collect_set(attribute) AS errors
         |FROM $errCellView
         |GROUP BY $rowId
       """.stripMargin)

    val repairBase = withTempView(errAttrDf) { errAttrView =>
      val cleanAttrs = spark.table(discreteAttrView).columns.map {
        case attr if attr == rowId =>
          s"$discreteAttrView.$rowId"
        case attr if attrsToRepair.contains(attr) =>
          s"IF(array_contains(errors, '$attr'), NULL, $attr) AS $attr"
        case cleanAttr =>
          cleanAttr
      }
      spark.sql(
        s"""
           |SELECT ${cleanAttrs.mkString(", ")}
           |FROM $discreteAttrView
           |LEFT OUTER JOIN $errAttrView
           |ON $discreteAttrView.$rowId = $errAttrView.$rowId
         """.stripMargin)
    }
    val repairBaseView = createAndCacheTempView(repairBase, "repair_base_cells")
    Seq("repair_base_cells" -> repairBaseView).asJson
  }

  private[python] def computeAttrStats(
      discreteAttrView: String,
      discreteAttrs: Seq[String],
      attrPairsToRepair: Seq[(String, String)],
      statSampleRatio: Double,
      statThreshold: Double): DataFrame = {
    assert(discreteAttrs.nonEmpty && attrPairsToRepair.nonEmpty)
    val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
    val inputDf = if (statSampleRatio < 1.0) {
      spark.table(discreteAttrView).sample(statSampleRatio)
    } else {
      spark.table(discreteAttrView)
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

  private def whereCaluseToFilterStat(a: String, attrs: Seq[String]): String = {
    s"$a IS NOT NULL AND ${attrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"
  }

  private[python] def compuatePairwiseAttrStats(
      rowCnt: Long,
      attrStatView: String,
      discreteAttrView: String,
      discreteAttrs: Seq[String],
      attrPairsToRepair: Seq[(String, String)],
      minCorrThres: Double): (Map[String, Seq[(String, Double)]], Map[String, Long]) = {
    assert(rowCnt > 0 && discreteAttrs.nonEmpty && attrPairsToRepair.nonEmpty)
    // Computes the conditional entropy: H(x|y) = H(x,y) - H(y).
    // H(x,y) denotes H(x U y). If H(x|y) = 0, then y determines x, i.e., y -> x.
    val hXYs = withJobDescription("compute conditional entropy H(x|y)") {
      val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
      pairSets.map { attrPairKey =>
        attrPairKey -> {
          val Seq(a1, a2) = attrPairKey.toSeq
          val df = spark.sql(
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
          val df = spark.sql(
            s"""
               |SELECT -SUM(hY) hY
               |FROM (
               |  /* TODO: Needs to reconsider how-to-handle NULL */
               |  /* Use `MAX` to drop ($attrKey, null) tuples in `$discreteAttrView` */
               |  SELECT $attrKey Y, (MAX(cnt) / $rowCnt) * log10(MAX(cnt) / $rowCnt) hY
               |  FROM $attrStatView
               |  WHERE ${whereCaluseToFilterStat(attrKey, discreteAttrs)}
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
    val pairwiseStats = attrPairsToRepair.map { case attrPair @ (rvX, rvY) =>
      // The conditional entropy is 0 for strongly correlated attributes and 1 for completely independent
      // attributes. We reverse this to reflect the correlation.
      val domainSize = domainStatMap(rvX)
      attrPair -> (1.0 - ((hXYs(Set(rvX, rvY)) - hYs(rvY)) / scala.math.log10(domainSize)))
    }

    val pairwiseStatMap = pairwiseStats.groupBy { case ((attrToRepair, _), _) =>
      attrToRepair
    }.map { case (k, v) =>
      k -> v.map { case ((_, attr), v) =>
        (attr, v)
      }.sortBy(_._2).reverse
    }
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

    (pairwiseStatMap, domainStatMap)
  }

  private[python] def computeCorrAttrs(
      pairwiseStatMap: Map[String, Seq[(String, Double)]],
      maxAttrsToComputeDomains: Int,
      minCorrThres: Double): Map[String, Seq[(String, Double)]] = {
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
      discreteAttrView: String,
      errCellView: String,
      rowId: String,
      continuousAttrList: String,
      targetAttrList: String,
      discreteAttrList: String,
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
      s"discreteAttrView=$discreteAttrView errCellView=$errCellView rowId=$rowId " +
      s"continousAttrList=${if (continuousAttrList.nonEmpty) continuousAttrList else "<none>"} " +
      s"targetAttrList=$targetAttrList discreteAttrList=$discreteAttrList " +
      s"rowCount=$rowCount maxAttrsToComputeDomains=$maxAttrsToComputeDomains " +
      s"statSampleRatio=$statSampleRatio statThreshold=$statThreshold minCorrThres=$minCorrThres " +
      s"domain_threshold=alpha:$domain_threshold_alpha,beta=$domain_threshold_beta")

    val discreteAttrs = SparkUtils.stringToSeq(discreteAttrList)
    val attrsToRepair = SparkUtils.stringToSeq(targetAttrList)

    assert(checkSchema(errCellView, "attribute STRING, current_value STRING", rowId, strict = true))
    assert(discreteAttrs.nonEmpty && attrsToRepair.nonEmpty)

    val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
      discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
    }
    val attrStatDf = computeAttrStats(
      discreteAttrView, discreteAttrs, attrPairsToRepair, statSampleRatio, statThreshold)

    withTempView(attrStatDf, cache = true) { attrStatView =>
      val (pairwiseStatMap, domainStatMap) = compuatePairwiseAttrStats(
        rowCount, attrStatView, discreteAttrView, discreteAttrs,
        attrPairsToRepair, minCorrThres)
      val corrAttrs = computeCorrAttrs(pairwiseStatMap, maxAttrsToComputeDomains, minCorrThres)
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
               |  $discreteAttrView l, $repairCellView r
               |WHERE
               |  l.$rowId = r.$rowId
             """.stripMargin)
        }
      } else {
        // Needs to keep the correlated attributes for selecting their domains
        val corrAttrSet = corrAttrs.flatMap(_._2.map(_._1)).toSet
        val corrCols = if (corrAttrSet.nonEmpty) {
          corrAttrSet.mkString(", ", ", ", "")
        } else {
          ""
        }

        val rvDf = withTempView(repairCellDf) { repairCellView =>
          spark.sql(
            s"""
               |SELECT
               |  l.$rowId, r.attribute, r.current_value $corrCols
               |FROM
               |  $discreteAttrView l, $repairCellView r
               |WHERE
               |  l.$rowId = r.$rowId
             """.stripMargin)
        }

        withTempView(rvDf) { rvView =>
          val continousAttrs = SparkUtils.stringToSeq(continuousAttrList).toSet
          corrAttrs.map { case (attribute, corrAttrsWithScores) =>
            // Adds an empty domain for initial state
            val initDomainDf = spark.sql(
              s"""
                 |SELECT $rowId, attribute, current_value, $domainInitValue domain $corrCols
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
                   |    WHERE ${whereCaluseToFilterStat(attribute, discreteAttrs)}
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
      // Checks if # of rows in `cellDomainView` is the same with # of error cells
      assert(cellDomainDf.count == repairCellDf.count)
      Seq("cell_domain" -> cellDomainView,
        "pairwise_attr_stats" -> pairwiseStatMap.mapValues(seqToJson)
      ).asJson
    }
  }
}
