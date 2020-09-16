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

import org.apache.spark.SparkException
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.util.{Utils => SparkUtils}

object ScavengerMiscApi extends BaseScavengerRepairApi {

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

  private def compute2gram(ar: Seq[String]): Seq[String] = {
    if (ar != null) {
      val buffer = mutable.ArrayBuffer[String]()
      ar.foreach { str =>
        if (str != null) {
          if (str.length > 2) {
            for (i <- 0 to str.length - 2) {
              buffer += s"${str.charAt(i)}${str.charAt(i + 1)}"
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
   * To reduce the time complexity of data cleaning, there is the well-known pre-processing
   * technique, so called "blocking" [11][12], that split input rows into multiple blocks.
   * Then, data cleaning is applied into blocks independently. This method splits
   * an input `dbName`.`tableName` into `k` blocks.
   */
  def blockRows(dbName: String, tableName: String, rowId: String, targetAttrList: String, k: Int): DataFrame = {
    logBasedOnLevel(s"blockSimilarRows called with: dbName=$dbName tableName=$tableName rowId=$rowId" +
      s"targetAttrList=$targetAttrList, k=$k")

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
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

      // Since we assume input rows have lexical heterogeneity (e.g., lexical errors like typos),
      // we compute bi-gram for the rows and create features by using bag-of-bigram.
      val featureAttr = getRandomString()
      val featureDf = {
        import sparkSession.implicits._
        val compute2gramUdf = udf[Seq[String], Seq[String]](compute2gram)
        val Seq(inputAttr, bigramAttr) = Seq.fill(2)(0).map {
          _ => getRandomString()
        }
        val df = inputDf.selectExpr(rowId, s"array(${targetAttrs.mkString(", ")}) AS $inputAttr")
          .withColumn(bigramAttr, compute2gramUdf($"$inputAttr"))
          .drop(inputAttr)

        val cv = new CountVectorizer().setInputCol(bigramAttr).setOutputCol(featureAttr)
        cv.fit(df).transform(df).drop(bigramAttr)
      }

      val bkm = new BisectingKMeans()
        .setFeaturesCol(featureAttr)
        .setPredictionCol("k")
        .setK(k)
        .setSeed(0)

      bkm.fit(featureDf).transform(featureDf).drop(featureAttr)
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
