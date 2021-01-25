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

import scala.io.Source

import org.apache.spark.python.DenialConstraints
import org.apache.spark.sql._
import org.apache.spark.sql.types.StringType
import org.apache.spark.util.LoggingBasedOnLevel
import org.apache.spark.util.RepairUtils._

object ErrorDetectorApi extends LoggingBasedOnLevel {

  def detectNullCells(qualifiedName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"detectNullCells called with: qualifiedName=$qualifiedName rowId=$rowId")
    NullErrorDetector.detect(qualifiedName, rowId)
  }

  def detectErrorCellsFromRegEx(
      qualifiedName: String,
      rowId: String,
      regex: String,
      cellsAsString: Boolean = false): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromRegEx called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId, regex=$regex, cellAsString=$cellsAsString")
    RegExErrorDetector.detect(qualifiedName, rowId,
      Map("regex" -> regex, "cellsAsString" -> cellsAsString))
  }

  def detectErrorCellsFromConstraints(
      qualifiedName: String,
      rowId: String,
      constraintFilePath: String): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromConstraints called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId constraintFilePath=$constraintFilePath")
    ConstraintErrorDetector.detect(qualifiedName, rowId,
      Map("constraintFilePath" -> constraintFilePath))
  }

  def detectErrorCellsFromOutliers(
      qualifiedName: String,
      rowId: String,
      approxEnabled: Boolean = false): DataFrame = {
    logBasedOnLevel(s"detectErrorCellsFromOutliers called with: qualifiedName=$qualifiedName " +
      s"rowId=$rowId approxEnabled=$approxEnabled")
    GaussianOutlierErrorDetector.detect(qualifiedName, rowId,
      Map("approxEnabled" -> approxEnabled))
  }
}

abstract class ErrorDetector extends RepairBase {

  def detect(qualifiedName: String, rowId: String, options: Map[String, Any]): DataFrame

  protected def emptyTable(df: DataFrame, rowId: String): DataFrame = {
    val rowIdType = df.schema.find(_.name == rowId).get.dataType.sql
    createEmptyTable(s"$rowId $rowIdType, attribute STRING")
  }

  protected def getOptionValue[T](key: String, options: Map[String, Any]): T = {
    assert(options.contains(key))
    options(key).asInstanceOf[T]
  }

  protected def loggingErrorStats(
      detectorIdent: String,
      inputName: String,
      errCellView: String,
      attrsToRepair: Seq[String]): Unit = {

    logBasedOnLevel({
      logBasedOnLevel({
        val errorNumOfEachAttribute = {
          val df = spark.sql(s"SELECT attribute, COUNT(1) FROM $errCellView GROUP BY attribute")
          df.collect.map { case Row(attribute: String, n: Long) => s"$attribute:$n" }
        }
        s"""
           |$detectorIdent found errors:
           |  ${errorNumOfEachAttribute.mkString("\n  ")}
         """.stripMargin
      })

      val inputDf = spark.table(inputName)
      val tableAttrs = inputDf.schema.map(_.name)
      val tableAttrNum = tableAttrs.length
      val tableRowCnt = inputDf.count()
      val errCellNum = spark.table(errCellView).count()
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
      options: Map[String, Any] = Map.empty): DataFrame = {

    val inputDf = spark.table(qualifiedName)

    withTempView(inputDf, cache = true) { inputView =>
      // Detects error erroneous cells in a given table
      val sqls = inputDf.columns.filter(_ != rowId).map { attr =>
        s"""
           |SELECT $rowId, '$attr' AS attribute
           |FROM $inputView
           |WHERE $attr IS NULL
         """.stripMargin
      }

      val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))

      withTempView(errCellDf) { errCellView =>
        val attrsToRepair = {
          spark.sql(s"SELECT collect_set(attribute) FROM $errCellView")
            .collect.head.getSeq[String](0)
        }
        loggingErrorStats(
          "NULL-based error detector",
          qualifiedName,
          errCellView,
          attrsToRepair)
      }
      errCellDf
    }
  }
}

object RegExErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      options: Map[String, Any] = Map.empty): DataFrame = {

    val inputDf = spark.table(qualifiedName)

    // If `regex` not given, just returns an empty table
    val regex = getOptionValue[String]("regex", options)
    if (regex == null || regex.trim.isEmpty) {
      emptyTable(inputDf, rowId)
    } else {
      withTempView(inputDf, cache = true) { inputView =>
        // Detects error erroneous cells in a given table
        val cellsAsString = getOptionValue[Boolean]("cellsAsString", options)
        val sqls = if (cellsAsString) {
          inputDf.columns.filter(_ != rowId).map { attr =>
            s"""
               |SELECT $rowId, '$attr' AS attribute
               |FROM $inputView
               |WHERE CAST($attr AS STRING) RLIKE '$regex'
             """.stripMargin
          }.toSeq
        } else {
          inputDf.schema.filter { f => f.name != rowId && f.dataType == StringType }.map { f =>
            s"""
               |SELECT $rowId, '${f.name}' AS attribute
               |FROM $inputView
               |WHERE ${f.name} RLIKE '$regex'
             """.stripMargin
          }
        }

        if (sqls.isEmpty) {
          emptyTable(inputDf, rowId)
        } else {
          val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))

          withTempView(errCellDf) { errCellView =>
            val attrsToRepair = {
              spark.sql(s"SELECT collect_set(attribute) FROM $errCellView")
                .collect.head.getSeq[String](0)
            }
            loggingErrorStats(
              "RegEx-based error detector",
              qualifiedName,
              errCellView,
              attrsToRepair)
          }
          errCellDf
        }
      }
    }
  }
}

object ConstraintErrorDetector extends ErrorDetector {

  override def detect(
      qualifiedName: String,
      rowId: String,
      options: Map[String, Any] = Map.empty): DataFrame = {

    val inputDf = spark.table(qualifiedName)

    // If `constraintFilePath` not given, just returns an empty table
    val constraintFilePath = getOptionValue[String]("constraintFilePath", options)
    if (constraintFilePath == null || constraintFilePath.trim.isEmpty) {
      emptyTable(inputDf, rowId)
    } else {
      withTempView(inputDf, cache = true) { inputView =>
        // Loads all the denial constraints from a given file path
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

        if (constraints.predicates.isEmpty) {
          emptyTable(inputDf, rowId)
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
          val sqls = constraints.predicates.map { preds =>
            import DenialConstraints._
            val attrs = preds.flatMap(_.references).distinct
            // TODO: Needs to look for a more smart logic to filter error cells
            s"""
               |SELECT DISTINCT $rowId, explode(array(${attrs.map(a => s"'$a'").mkString(",")})) attribute
               |FROM (
               |  SELECT $leftRelationIdent.$rowId FROM $inputView AS $leftRelationIdent
               |  WHERE EXISTS (
               |    SELECT $rowId FROM $inputView AS $rightRelationIdent
               |    WHERE ${preds.mkString(" AND ")}
               |  )
               |)
             """.stripMargin
          }

          val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))
          val errCellView = createAndCacheTempView(errCellDf)
          loggingErrorStats(
            "Constraint-based error detector",
            qualifiedName,
            errCellView,
            constraints.references
          )
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
      options: Map[String, Any] = Map.empty): DataFrame = {

    val inputDf = spark.table(qualifiedName)

    val continousAttrs = inputDf.schema
      .filter(f => continousTypes.contains(f.dataType)).map(_.name)
    if (continousAttrs.isEmpty) {
      emptyTable(inputDf, rowId)
    } else {
      val percentileExprs = continousAttrs.map { attr =>
        val approxEnabled = getOptionValue[Boolean]("approxEnabled", options)
        if (approxEnabled) {
          s"percentile_approx($attr, array(0.25, 0.75), 1000)"
        } else {
          s"percentile($attr, array(0.25, 0.75))"
        }
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
           |SELECT $rowId, '$attr' attribute
           |FROM $qualifiedName
           |WHERE $attr < $lower OR $attr > $upper
         """.stripMargin
      }

      val errCellDf = spark.sql(sqls.mkString(" UNION ALL "))
      val errCellView = createAndCacheTempView(errCellDf)
      loggingErrorStats(
        "Outlier-based error detector",
        qualifiedName,
        errCellView,
        continousAttrs
      )
      errCellDf
    }
  }
}
