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

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import org.apache.spark.python.DenialConstraints
import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql._
import org.apache.spark.sql.types.StringType
import org.apache.spark.util.LoggingBasedOnLevel
import org.apache.spark.util.RepairUtils._
import org.apache.spark.util.{Utils => SparkUtils}

object ErrorDetectorApi extends LoggingBasedOnLevel {

  def detectNullCells(qualifiedName: String, rowId: String, targetAttrList: String): DataFrame = {
    logBasedOnLevel(s"detectNullCells called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId targetAttrList=$targetAttrList")
    NullErrorDetector.detect(qualifiedName, rowId, SparkUtils.stringToSeq(targetAttrList))
  }

  def detectErrorCellsFromRegEx(
      qualifiedName: String,
      rowId: String,
      targetAttrList: String,
      attr: String,
      regex: String): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromRegEx called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId targetAttrList=$targetAttrList regex=$regex")
    RegExErrorDetector.detect(qualifiedName, rowId, SparkUtils.stringToSeq(targetAttrList),
      Map("attr" -> attr, "regex" -> regex))
  }

  def detectErrorCellsFromConstraints(
      qualifiedName: String,
      rowId: String,
      targetAttrList: String,
      constraintFilePath: String): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromConstraints called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId targetAttrlist=$targetAttrList constraintFilePath=$constraintFilePath")
    ConstraintErrorDetector.detect(qualifiedName, rowId, SparkUtils.stringToSeq(targetAttrList),
      Map("constraintFilePath" -> constraintFilePath))
  }

  def detectErrorCellsFromOutliers(
      qualifiedName: String,
      rowId: String,
      targetAttrList: String,
      approxEnabled: Boolean = false): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromOutliers called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId targetAttrList=$targetAttrList approxEnabled=$approxEnabled")
    GaussianOutlierErrorDetector.detect(qualifiedName, rowId, SparkUtils.stringToSeq(targetAttrList),
      Map("approxEnabled" -> approxEnabled))
  }
}

abstract class ErrorDetector extends RepairBase {

  def detect(
    qualifiedName: String,
    rowId: String,
    targetAttrs: Seq[String],
    options: Map[String, Any]): DataFrame

  protected def createEmptyResultDfFrom(df: DataFrame, rowId: String): DataFrame = {
    val rowIdType = df.schema.find(_.name == rowId).get.dataType.sql
    createEmptyTable(s"`$rowId` $rowIdType, attribute STRING")
  }

  protected def getInput(qualifiedName: String, targetAttrs: Seq[String]): (DataFrame, Seq[String]) = {
    val df = spark.table(qualifiedName)
    val targetColumns = if (targetAttrs.nonEmpty) {
      val filteredColumns = df.columns.filter(targetAttrs.contains)
      assert(filteredColumns.nonEmpty, s"Target attributes not found in $qualifiedName: ${targetAttrs.mkString(",")}")
      filteredColumns
    } else {
      df.columns
    }
    (df, targetColumns.toSeq)
  }

  protected def getOptionValue[T](key: String, options: Map[String, Any]): T = {
    assert(options.contains(key))
    options(key).asInstanceOf[T]
  }

  protected def loggingErrorStats(
      detectorIdent: String,
      inputName: String,
      errCellDf: DataFrame): Unit = {

    lazy val attrsToRepair = ArrayBuffer[String]()

    logBasedOnLevel({
      withTempView(errCellDf) { errCellView =>
        val errorNumOfEachAttribute = {
          val df = spark.sql(s"SELECT attribute, COUNT(1) FROM $errCellView GROUP BY attribute")
          df.collect.map { case Row(attribute: String, n: Long) =>
            attrsToRepair += attribute
            s"$attribute:$n"
          }
        }
        s"""
           |$detectorIdent found errors:
           |  ${errorNumOfEachAttribute.mkString("\n  ")}
         """.stripMargin
      }
    })
    logBasedOnLevel({
      val inputDf = spark.table(inputName)
      val tableAttrs = inputDf.schema.map(_.name)
      val tableAttrNum = tableAttrs.length
      val tableRowCnt = inputDf.count()
      val errCellNum = errCellDf.count()
      val totalCellNum = tableRowCnt * tableAttrNum
      val errRatio = (errCellNum + 0.0) / totalCellNum
      s"$detectorIdent found $errCellNum/$totalCellNum error cells (${errRatio * 100.0}%) of " +
        s"${attrsToRepair.size}/${tableAttrs.size} attributes (${attrsToRepair.mkString(",")}) " +
        s"in the input '$inputName'"
    })
  }
}

object NullErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      targetAttrs: Seq[String],
      options: Map[String, Any] = Map.empty): DataFrame = {

    val (inputDf, inputColumns) = getInput(qualifiedName, targetAttrs)

    withTempView(inputDf, cache = true) { inputView =>
      // Detects error erroneous cells in a given table
      val sqls = inputColumns.filter(_ != rowId).map { attr =>
        s"""
           |SELECT `$rowId`, '$attr' AS attribute
           |FROM $inputView
           |WHERE `$attr` IS NULL
         """.stripMargin
      }

      val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))
      loggingErrorStats("NULL-based error detector", qualifiedName, errCellDf)
      errCellDf
    }
  }
}

object RegExErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      targetAttrs: Seq[String],
      options: Map[String, Any] = Map.empty): DataFrame = {

    val (inputDf, inputColumns) = getInput(qualifiedName, targetAttrs)

    val targetColumn = getOptionValue[String]("attr", options)
    val regex = getOptionValue[String]("regex", options)
    if (!inputColumns.contains(targetColumn) || regex == null || regex.trim.isEmpty) {
      createEmptyResultDfFrom(inputDf, rowId)
    } else {
      withTempView(inputDf) { inputView =>
        val errCellDf = spark.sql(
          s"""
             |SELECT `$rowId`, '$targetColumn' AS attribute
             |FROM $inputView
             |WHERE CAST(`$targetColumn` AS STRING) NOT RLIKE '$regex' OR `$targetColumn` IS NULL
           """.stripMargin)

        loggingErrorStats("RegEx-based error detector", qualifiedName, errCellDf)
        errCellDf
      }
    }
  }
}

object ConstraintErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      targetAttrs: Seq[String],
      options: Map[String, Any] = Map.empty): DataFrame = {

    val (inputDf, inputColumns) = getInput(qualifiedName, targetAttrs)

    // If `constraintFilePath` not given, just returns an empty table
    val constraintFilePath = getOptionValue[String]("constraintFilePath", options)
    if (constraintFilePath == null || constraintFilePath.trim.isEmpty) {
      createEmptyResultDfFrom(inputDf, rowId)
    } else {
      withTempView(inputDf, cache = true) { inputView =>
        // Loads all the denial constraints from a given file path
        var file: Source = null
        val constraints = try {
          file = Source.fromFile(new URI(constraintFilePath).getPath)
          file.getLines()
          DenialConstraints.parseAndVerifyConstraints(file.getLines(), qualifiedName, inputDf.columns.toSeq)
        } finally {
          if (file != null) {
            file.close()
          }
        }

        if (constraints.predicates.isEmpty) {
          createEmptyResultDfFrom(inputDf, rowId)
        } else {
          logBasedOnLevel({
            val constraintLists = constraints.predicates.zipWithIndex.map { case (preds, i) =>
              preds.map(_.toString).mkString(s" [$i] ", ",", "")
            }
            s"""
               |Loads constraints from '$constraintFilePath':
               |${constraintLists.mkString("\n")}
             """.stripMargin
          })

          // Detects error erroneous cells in a given table
          val sqls = constraints.predicates.flatMap { preds =>
            import DenialConstraints._
            val attrs = preds.flatMap(_.references).filter(inputColumns.contains).distinct
            if (attrs.nonEmpty) {
              // TODO: Needs to look for a more smart logic to filter error cells
              Some(s"""
                 |SELECT DISTINCT `$rowId`, explode(array(${attrs.map(a => s"'$a'").mkString(",")})) attribute
                 |FROM (
                 |  SELECT $leftRelationIdent.`$rowId` FROM $inputView AS $leftRelationIdent
                 |  WHERE EXISTS (
                 |    SELECT `$rowId` FROM $inputView AS $rightRelationIdent
                 |    WHERE ${preds.mkString(" AND ")}
                 |  )
                 |)
               """.stripMargin)
            } else {
              None
            }
          }

          val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))
          loggingErrorStats("Constraint-based error detector", qualifiedName, errCellDf)
          errCellDf
        }
      }
    }
  }
}

// TODO: Needs to support more sophisticated outlier detectors, e.g., a nonparametric histogram
// approach and a correlation based approach (named 'OD' in the HoloDetect paper [1]).
// We might be able to compute outliers by reusing [[RepairApi.computeDomainInErrorCells]].
object GaussianOutlierErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      targetAttrs: Seq[String],
      options: Map[String, Any] = Map.empty): DataFrame = {

    val (inputDf, inputColumns) = getInput(qualifiedName, targetAttrs)

    val continousAttrs = inputDf.schema.filter { f =>
      inputColumns.contains(f.name) && continousTypes.contains(f.dataType)
    }.map(_.name)

    if (continousAttrs.isEmpty) {
      createEmptyResultDfFrom(inputDf, rowId)
    } else {
      val percentileExprs = continousAttrs.map { attr =>
        val approxEnabled = getOptionValue[Boolean]("approxEnabled", options)
        val expr = if (approxEnabled) {
          s"percentile_approx($attr, array(0.25, 0.75), 1000)"
        } else {
          s"percentile($attr, array(0.25, 0.75))"
        }
        s"CAST($expr AS ARRAY<DOUBLE>) $attr"
      }

      val percentileRow = spark.sql(
        s"""
           |SELECT ${percentileExprs.mkString(", ")}
           |FROM $qualifiedName
         """.stripMargin).collect.head

      val sqls = continousAttrs.zipWithIndex.map { case (attr, i) =>
        // Detects outliers simply based on a Box-and-Whisker plot
        // TODO: Needs to support more sophisticated ways to detect outliers
        val Seq(q1, q3) = percentileRow.getSeq[Double](i)
        val (lower, upper) = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        logBasedOnLevel(s"Non-outlier values in $attr should be in [$lower, $upper]")
        s"""
           |SELECT `$rowId`, '$attr' attribute
           |FROM $qualifiedName
           |WHERE `$attr` < $lower OR `$attr` > $upper
         """.stripMargin
      }

      val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))
      loggingErrorStats("Outlier-based error detector", qualifiedName, errCellDf)
      errCellDf
    }
  }
}
