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

import org.apache.spark.sql._

object ScavengerErrorDetectorApi extends BaseScavengerRepairApi {

  private def loggingErrorStats(
      detectorIdent: String,
      inputName: String,
      errCellView: String,
      attrsToRepair: Seq[String]): Unit = {
    logBasedOnLevel({
      val spark = activeSparkSession()
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

  def detectNullCells(dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"detectNullCells called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)

      withTempView(inputDf, cache = true) { inputView =>
        // Detects error erroneous cells in a given table
        val sqls = inputDf.columns.map { attr =>
          s"""
             |SELECT $rowId, '$attr' AS attribute
             |FROM $inputView
             |WHERE $attr IS NULL
           """.stripMargin
        }

        val errCellDf = sparkSession.sql(sqls.mkString(" UNION ALL "))

        withTempView(errCellDf) { errCellView =>
          val attrsToRepair = {
            sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
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

  def detectErrorCellsFromConstraints(
      constraintFilePath: String,
      dbName: String,
      tableName: String,
      rowId: String): DataFrame = {

    logBasedOnLevel(s"detectErrorCellsFromConstraints called with: constraintFilePath=$constraintFilePath " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
      lazy val emptyTable = {
        val rowIdType = inputDf.schema.find(_.name == rowId).get.dataType.sql
        createEmptyTable(s"$rowId $rowIdType, attribute STRING")
      }

      // If `constraintFilePath` not given, just returns an empty table
      if (constraintFilePath == null || constraintFilePath.trim.isEmpty) {
        emptyTable
      } else {
        withTempView(inputDf, cache = true) { inputView =>
          val constraints = loadConstraintsFromFile(constraintFilePath, tableName, inputDf.columns)
          if (constraints.predicates.isEmpty) {
            emptyTable
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
              val attrs = preds.flatMap(_.references)
              s"""
                 |SELECT $rowId, explode(array(${attrs.map(a => s"'$a'").mkString(",")})) attribute
                 |FROM (
                 |  SELECT t1.$rowId
                 |  FROM $inputView AS ${constraints.leftTable}
                 |  WHERE EXISTS (
                 |    SELECT $rowId
                 |    FROM $inputView AS ${constraints.rightTable}
                 |    WHERE ${preds.mkString(" AND ")}
                 |  )
                 |)
               """.stripMargin
            }

            val errCellDf = sparkSession.sql(sqls.mkString(" UNION ALL "))
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
  // We might be able to compute outliers by reusing [[ScavengerRepairApi.computeDomainInErrorCells]].
  def detectErrorCellsFromOutliers(
      dbName: String,
      tableName: String,
      rowId: String,
      approxEnabled: Boolean): DataFrame = {

    logBasedOnLevel(s"detectErrorCellsFromOutliers called with: dbName=$dbName tableName=$tableName " +
      s"rowId=$rowId approxEnabled=$approxEnabled")

    withSparkSession { sparkSession =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName, rowId)
      lazy val emptyTable = {
        val rowIdType = inputDf.schema.find(_.name == rowId).get.dataType.sql
        createEmptyTable(s"$rowId $rowIdType, attribute STRING")
      }

      val continousAttrs = inputDf.schema
        .filter(f => continousTypes.contains(f.dataType)).map(_.name)
      if (continousAttrs.isEmpty) {
        emptyTable
      } else {
        val percentileExprs = continousAttrs.map { attr =>
          if (approxEnabled) {
            s"percentile_approx($attr, array(0.25, 0.75), 1000)"
          } else {
            s"percentile($attr, array(0.25, 0.75))"
          }
        }

        val percentileRow = sparkSession.sql(
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

        val errCellDf = sparkSession.sql(sqls.mkString(" UNION ALL "))
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
}
