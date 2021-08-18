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

import scala.collection.mutable
import scala.util.Try
import scala.collection.JavaConverters._

import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql.{DataFrame, Row, SparkCommandUtils}
import org.apache.spark.sql.catalyst.plans.logical.Histogram
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.util.RepairUtils._
import org.apache.spark.util.{Utils => SparkUtils}

object RepairMiscApi extends RepairBase {

  /**
   * To compare result rows easily, this method flattens an input table
   * as a schema (`rowId`, attribute, val).
   */
  def flattenTable(dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"flattenTable called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    val (inputDf, _) = checkAndGetQualifiedInputName(dbName, tableName, rowId)
    val expr = inputDf.schema.filter(_.name != rowId)
      .map { f => s"STRUCT($rowId, '${f.name}', CAST(${f.name} AS STRING))" }
      .mkString("ARRAY(", ", ", ")")
    inputDf.selectExpr(s"INLINE($expr) AS ($rowId, attribute, value)")
  }

  // TODO: Implements this func as an expression for supporting codegen
  private[python] def computeQgram(q: Int, ar: Seq[String]): Seq[String] = {
    require(q > 0, s"`q` must be positive, but $q got")
    if (ar != null) {
      val buffer = mutable.ArrayBuffer[String]()
      ar.foreach { str =>
        if (str != null) {
          if (str.length > q) {
            for (i <- 0 to str.length - q) {
              buffer += str.substring(i, i + q)
            }
          } else {
            buffer += str
          }
        }
      }
      buffer
    } else {
      Nil
    }
  }

  /**
   * To reduce the running time of data cleaning, this method splits an input `dbName`.`tableName`
   * into `k` groups so that each group can have similar rows. Then, data cleaning is
   * applied into the groups independently.
   */
  def splitInputTableInto(
      k: Int,
      dbName: String,
      tableName: String,
      rowId: String,
      targetAttrList: String,
      options: String): DataFrame = {
    logBasedOnLevel(s"splitInputTableInto called with: k=$k dbName=$dbName tableName=$tableName rowId=$rowId " +
      s"targetAttrList=$targetAttrList options=${if (options.nonEmpty) s"{$options}" else ""}")

    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName(dbName, tableName, rowId)
    val optionMap = SparkUtils.stringToSeq(options).map(_.split("=")).filter(_.length == 2)
      .map { case Array(k, v) => k -> v }.toMap
    val targetAttrs = if (targetAttrList.isEmpty) {
      inputDf.columns.filterNot(_ == rowId).toSeq
    } else {
      val attrs = SparkUtils.stringToSeq(targetAttrList)
      val unknownAttrs = attrs.filterNot(inputDf.columns.contains)
      if (unknownAttrs.nonEmpty) {
        throw AnalysisException(s"Columns '${unknownAttrs.mkString(", ")}' " +
          s"do not exist in '$qualifiedName'")
      }
      attrs
    }

    // There are many types of errors: typos, missing values, incorrect values, contradicting facts,
    // shifted data and so on. Therefore, We need to select a suitable approach so that
    // it could detect and handle given errors correctly. Since we assume, in this method,
    // input rows have lexical heterogeneity (e.g., lexical errors like typos),
    // we compute q-gram for the rows and create features by using bag-of-q-gram.
    val featureAttr = getRandomString()
    val featureDf = {
      val sparkSession = spark
      import sparkSession.implicits._
      val computeQgramUdf = udf[Seq[String], Int, Seq[String]](computeQgram)
      val Seq(inputAttr, bigramAttr) = Seq.fill(2)(0).map {
        _ => getRandomString()
      }
      val q = if (optionMap.contains("q")) Try(optionMap("q").toInt).getOrElse(2) else 2
      // TODO: Adds an option for using a feature hashing trick to reduce dimension
      // https://spark.apache.org/docs/latest/ml-features.html#featurehasher
      val df = inputDf.selectExpr(rowId, s"array(${targetAttrs.mkString(", ")}) AS $inputAttr")
        .withColumn(bigramAttr, computeQgramUdf(lit(q), $"$inputAttr"))
        .drop(inputAttr)

      val cv = new CountVectorizer().setInputCol(bigramAttr).setOutputCol(featureAttr)
      cv.fit(df).transform(df).drop(bigramAttr)
    }

    // Currently, the two types of the clustering algorithms ('bisect-kmeans' and 'kmeans++')
    // implemented in Spark MLlib are available here. For implementation details,
    // please check a document below:
    // https://spark.apache.org/docs/latest/ml-clustering.html#clustering
    def createKmeansPlusPlus() = new KMeans()
      .setFeaturesCol(featureAttr)
      .setPredictionCol("k")
      .setK(k)
      .setSeed(0)

    def createBisectingKmeans() = new BisectingKMeans()
      .setFeaturesCol(featureAttr)
      .setPredictionCol("k")
      .setK(k)
      .setSeed(0)

    val clusteringAlg = optionMap.get("clusteringAlg") match {
      case Some("bisect-kmeans") => createKmeansPlusPlus()
      case Some("kmeans++") => createBisectingKmeans()
      case Some(clsAlg) =>
        throw new IllegalArgumentException(s"Unknown clustering algorithm found: $clsAlg")
      case None =>
        logWarning("No clustering algorithm given, so use bisecting k-means by default")
        createBisectingKmeans()
    }
    clusteringAlg.fit(featureDf).transform(featureDf).drop(featureAttr)
  }

  def injectNullAt(
      dbName: String,
      tableName: String,
      targetAttrList: String,
      nullRatio: Double): DataFrame = {
    logBasedOnLevel(s"injectNullAt called with: dbName=$dbName tableName=$tableName " +
      s"targetAttrList=$targetAttrList, nullRatio=$nullRatio")

    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName(dbName, tableName)
    val targetAttrSet = if (targetAttrList.nonEmpty) {
      val attrs = SparkUtils.stringToSeq(targetAttrList)
      val unknownAttrs = attrs.filterNot(inputDf.columns.contains)
      if (unknownAttrs.nonEmpty) {
        throw AnalysisException(s"Columns '${unknownAttrs.mkString(", ")}' " +
          s"do not exist in '$qualifiedName'")
      }
      attrs.toSet
    } else {
      inputDf.columns.toSet
    }
    val exprs = inputDf.schema.map {
      case f if targetAttrSet.contains(f.name) =>
        s"IF(rand() > $nullRatio, ${f.name}, NULL) AS ${f.name}"
      case f =>
        f.name
    }
    inputDf.selectExpr(exprs: _*)
  }

  def repairAttrsFrom(
      repairUpdates: String,
      dbName: String,
      tableName: String,
      rowId: String): DataFrame = {
    logBasedOnLevel(s"repairAttrsFrom called with: repairUpdates=$repairUpdates " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    val (inputDf, _) = checkAndGetQualifiedInputName(dbName, tableName)

    // `repairedCells` must have `$rowId`, `attribute`, and `repaired` columns
    if (!checkSchema(repairUpdates, "attribute STRING, repaired STRING", rowId, strict = false)) {
      throw AnalysisException(s"Table '$repairUpdates' must have '$rowId', 'attribute', and 'repaired' columns")
    }

    val continousAttrTypeMap = inputDf.schema.filter(f => continousTypes.contains(f.dataType))
      .map { f => f.name -> f.dataType }.toMap
    val attrsToRepair = {
      spark.sql(s"SELECT collect_set(attribute) FROM $repairUpdates")
        .collect.head.getSeq[String](0).toSet
    }

    val repairDf = spark.sql(
      s"""
         |SELECT
         |  $rowId, map_from_entries(collect_list(r)) AS repairs
         |FROM (
         |  select $rowId, struct(attribute, repaired) r
         |  FROM $repairUpdates
         |)
         |GROUP BY
         |  $rowId
       """.stripMargin)

    withTempView(inputDf) { inputView =>
      withTempView(repairDf) { repairView =>
        val cleanAttrs = inputDf.schema.map {
          case f if f.name == rowId =>
            s"$inputView.$rowId"
          case f if attrsToRepair.contains(f.name) =>
            val repaired = if (continousAttrTypeMap.contains(f.name)) {
              val dataType = continousAttrTypeMap(f.name)
              if (integralTypes.contains(f.dataType)) {
                s"CAST(round(repairs['${f.name}']) AS ${dataType.catalogString})"
              } else {
                s"CAST(repairs['${f.name}'] AS ${dataType.catalogString})"
              }
            } else {
              s"repairs['${f.name}']"
            }
            s"if(array_contains(map_keys(repairs), '${f.name}'), $repaired, ${f.name}) AS ${f.name}"
          case f =>
            f.name
        }
        spark.sql(
          s"""
             |SELECT ${cleanAttrs.mkString(", ")}
             |FROM $inputView
             |LEFT OUTER JOIN $repairView
             |ON $inputView.$rowId = $repairView.$rowId
           """.stripMargin)
      }
    }
  }

  def computeAndGetStats(dbName: String, tableName: String, numBins: Int): DataFrame = {
    logBasedOnLevel(s"computeAndGetStats called with: dbName=$dbName tableName=$tableName")

    val (inputDf, _) = checkAndGetQualifiedInputName(dbName, tableName)
    val relation = inputDf.queryExecution.analyzed

    withSQLConf(SQLConf.HISTOGRAM_NUM_BINS.key -> s"$numBins") {
      def histogram2Seq(histOpt: Option[Histogram]): Option[Seq[Double]] = {
        histOpt.map { h =>
          val dist = h.bins.map { b => b.hi - b.lo }
          dist.map(_ / dist.sum)
        }
      }

      val (_, columnStats) = SparkCommandUtils.computeColumnStats(spark, relation, relation.output)
      val statRows = columnStats.map { case (attr, stat) =>
        Row.fromSeq(Seq(attr.name, stat.distinctCount.map(_.toLong), stat.min.map(_.toString),
          stat.max.map(_.toString), stat.nullCount.map(_.toLong), stat.avgLen,
          stat.maxLen, histogram2Seq(stat.histogram)))
      }
      // TODO: Add more metrics, e.g., central moments (See: `CentralMomentAgg`)
      val statScehma = StructType.fromDDL("attrName STRING, distinctCnt LONG, min STRING, " +
        "max STRING, nullCnt LONG, avgLen LONG, maxLen LONG, hist ARRAY<DOUBLE>")
      spark.createDataFrame(statRows.toSeq.asJava, statScehma)
    }
  }

  def convertToHistogram(targets: String, dbName: String, tableName: String): DataFrame = {
    logBasedOnLevel(s"convertToHistogram called with: targets=$targets" +
      s"dbName=$dbName tableName=$tableName")

    val (inputDf, _) = checkAndGetQualifiedInputName(dbName, tableName)
    val targetAttrSet = SparkUtils.stringToSeq(targets).toSet

    def isTarget(f: StructField): Boolean = {
      targetAttrSet.contains(f.name) && !continousTypes.contains(f.dataType)
    }

    withTempView(inputDf) { inputView =>
      val sqls = inputDf.schema.filter(isTarget).map { f =>
        s"""
           |SELECT '${f.name}' attribute, collect_list(b) histogram
           |FROM (
           |  SELECT named_struct('value', ${f.name}, 'cnt', COUNT(1)) b
           |  FROM $inputView
           |  WHERE ${f.name} IS NOT NULL
           |  GROUP BY ${f.name}
           |)
         """.stripMargin
      }
      spark.sql(sqls.mkString(" UNION ALL "))
    }
  }

  def toErrorMap(errCellView: String, dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"toErrorBitmap called with: errCellView=$errCellView " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    // `errCellView` must have `$rowId` and `attribute` columns
    if (!checkSchema(errCellView, "attribute STRING", rowId, strict = false)) {
      throw AnalysisException(s"Table '$errCellView' must have '$rowId' and 'attribute' columns")
    }

    val (inputDf, _) = checkAndGetQualifiedInputName(dbName, tableName, rowId)
    val attrsToRepair = {
      spark.sql(s"SELECT collect_set(attribute) FROM $errCellView")
        .collect.head.getSeq[String](0).toSet
    }

    val errorDf = spark.sql(
      s"""
         |SELECT
         |  $rowId, map_from_entries(COLLECT_LIST(r)) AS errors
         |FROM (
         |  select $rowId, struct(attribute, '') r
         |  FROM $errCellView
         |)
         |GROUP BY
         |  $rowId
       """.stripMargin)

    withTempView(inputDf) { inputView =>
      withTempView(errorDf) { errorView =>
        val errorBitmapExprs = inputDf.columns.flatMap {
          case attr if attrsToRepair.contains(attr) =>
            Some(s"IF(ISNOTNULL(errors['$attr']), '*', '-')")
          case attr if attr == rowId => None
          case _ => Some("'-'")
        }
        spark.sql(
          s"""
             |SELECT $inputView.$rowId, ${errorBitmapExprs.mkString(" || ")} AS error_map
             |FROM $inputView
             |LEFT OUTER JOIN $errorView
             |ON $inputView.$rowId = $errorView.$rowId
           """.stripMargin)
      }
    }
  }

  def generateDepGraph(
      path: String,
      dbName: String,
      tableName: String,
      format: String,
      targetAttrList: String,
      maxDomainSize: Int,
      maxAttrValueNum: Int,
      samplingRatio: Double,
      minCorrThres: Double,
      edgeLabel: Boolean): Unit = {
    logBasedOnLevel(s"generateDepGraph called with: path=$path dbName=$dbName tableName=$tableName " +
      s"format=$format targetAttrList=${if (targetAttrList.nonEmpty) targetAttrList else "<none>"} " +
      s"maxDomainSize=$maxDomainSize maxAttrValueNum=$maxAttrValueNum " +
      s"samplingRatio=$samplingRatio minCorrThres=$minCorrThres edgeLabel=$edgeLabel")

    val (inputDf, inputTable) = checkAndGetQualifiedInputName(dbName, tableName)
    val targetAttrs = if (targetAttrList.nonEmpty) {
      SparkUtils.stringToSeq(targetAttrList)
    } else {
      inputDf.columns.toSeq
    }
    DepGraphApi.generateDepGraph(
      path, inputTable, format, targetAttrs, maxDomainSize, maxAttrValueNum,
      samplingRatio, minCorrThres, edgeLabel)
  }
}