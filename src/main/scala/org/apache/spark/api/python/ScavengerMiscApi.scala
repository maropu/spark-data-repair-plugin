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

import org.apache.spark.SparkException
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.catalyst.plans.logical.{Histogram, LeafNode}
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.ScavengerUtils._
import org.apache.spark.util.{Utils => SparkUtils}

object ScavengerMiscApi extends ScavengerBase {

  /**
   * To compare result rows easily, this method flattens an input table
   * as a schema (`rowId`, attribute, val).
   */
  def flattenTable(dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"flattenTable called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { _ =>
      val (inputDf, _) = checkAndGetInputTable(dbName, tableName, rowId)
      val expr = inputDf.schema.filter(_.name != rowId)
        .map { f => s"STRUCT($rowId, '${f.name}', CAST(${f.name} AS STRING))" }
        .mkString("ARRAY(", ", ", ")")
      inputDf.selectExpr(s"INLINE($expr) AS ($rowId, attribute, value)")
    }
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

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
      val optionMap = SparkUtils.stringToSeq(options).map(_.split("=")).filter(_.length == 2)
        .map { case Array(k, v) => k -> v }.toMap
      val targetAttrs = if (targetAttrList.isEmpty) {
        inputDf.columns.filterNot(_ == rowId).toSeq
      } else {
        val attrs = SparkUtils.stringToSeq(targetAttrList)
        val unknownAttrs = attrs.filterNot(inputDf.columns.contains)
        if (unknownAttrs.nonEmpty) {
          throw new SparkException(s"Columns '${unknownAttrs.mkString(", ")}' " +
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
  }

  def injectNullAt(dbName: String, tableName: String, targetAttrList: String, nullRatio: Double): DataFrame = {
    logBasedOnLevel(s"injectNullAt called with: dbName=$dbName tableName=$tableName " +
      s"targetAttrList=$targetAttrList, nullRatio=$nullRatio")

    withSparkSession { _ =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName)
      val targetAttrSet = if (targetAttrList.nonEmpty) {
        val attrs = SparkUtils.stringToSeq(targetAttrList)
        val unknownAttrs = attrs.filterNot(inputDf.columns.contains)
        if (unknownAttrs.nonEmpty) {
          throw new SparkException(s"Columns '${unknownAttrs.mkString(", ")}' " +
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
  }

  def computeAndGetStats(dbName: String, tableName: String): DataFrame = {
    logBasedOnLevel(s"computeAndGetStats called with: dbName=$dbName tableName=$tableName")

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName)
      sparkSession.table(qualifiedName).cache()
      withSQLConf(
          SQLConf.PLAN_STATS_ENABLED.key -> "true",
          SQLConf.HISTOGRAM_ENABLED.key -> "true",
          SQLConf.HISTOGRAM_NUM_BINS.key -> "8") {

        sparkSession.sql(
          s"""
             |ANALYZE TABLE $qualifiedName COMPUTE STATISTICS
             |FOR ALL COLUMNS
           """.stripMargin)

        val tableStats = {
          val tableNode = inputDf.queryExecution.optimizedPlan.collectLeaves().head.asInstanceOf[LeafNode]
          val stat = tableNode.computeStats()
          assert(stat.attributeStats.nonEmpty, "stats must be computed")
          stat
        }

        def histogram2Seq(histOpt: Option[Histogram]): Option[Seq[Double]] = {
          histOpt.map { h =>
            val dist = h.bins.map { b => b.hi - b.lo }
            val maxValue = dist.max
            dist.map(_ / maxValue)
          }
        }
        val statNames = Seq(Seq("distinctCnt", "mix", "max", "nullCnt", "avgLen", "maxLen", "hist"))
        val statSeq = (statNames ++ tableStats.attributeStats.map { case (_, stat) =>
          Seq(stat.distinctCount, stat.min, stat.max, stat.nullCount, stat.avgLen,
            stat.maxLen, histogram2Seq(stat.histogram)).map(_.map(_.toString).orNull)
        }).transpose

        val statRows = statSeq.map(Row.fromSeq).asJava
        val statSchema = ("summary" +: tableStats.attributeStats.map(_._1.name).toSeq)
          .map { n => s"$n STRING" }.mkString(", ")
        sparkSession.createDataFrame(statRows, StructType.fromDDL(statSchema))
      }
    }
  }

  def toErrorMap(errCellView: String, dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"toErrorBitmap called with: errCellView=$errCellView " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      // `errCellView` must have `$rowId` and `attribute` columns
      checkIfColumnsExistIn(errCellView, rowId :: "attribute" :: Nil)

      val (inputDf, _) = checkAndGetInputTable(dbName, tableName, rowId)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
          .collect.head.getSeq[String](0).toSet
      }

      val errorDf = sparkSession.sql(
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
        withTempView(errorDf) { repairView =>
          val errorBitmapExprs = inputDf.columns.map {
            case attr if attrsToRepair.contains(attr) =>
              s"IF(ISNOTNULL(errors['$attr']), '*', '-')"
            case _ =>
              "'-'"
          }
          sparkSession.sql(
            s"""
               |SELECT ${errorBitmapExprs.mkString(" || ")} AS error_map
               |FROM $inputView
               |LEFT OUTER JOIN $repairView
               |ON $inputView.$rowId = $repairView.$rowId
             """.stripMargin)
        }
      }
    }
  }
}
