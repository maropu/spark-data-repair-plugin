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

import org.apache.spark.python._
import org.apache.spark.sql._

object ScavengerErrorDetectorApi extends BaseScavengerRepairApi {

  private def loggingErrorStats(
      detectorIdent: String,
      inputName: String,
      errCellView: String,
      attrsToRepair: Seq[String]): Unit = {
    assert(SparkSession.getActiveSession.nonEmpty)
    logBasedOnLevel({
      val spark = SparkSession.getActiveSession.get
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
        val errCellDf = inputDf.columns.map { attr =>
          sparkSession.sql(
            s"""
               |SELECT $rowId, '$attr' AS attribute
               |FROM $inputView
               |WHERE $attr IS NULL
             """.stripMargin)

        }.reduce(_.union(_)).cache()

        withTempView(errCellDf) { errCellView =>
          val attrsToRepair = {
            sparkSession.sql(s"SELECT collect_set(attribute) FROM $errCellView")
              .collect.head.getSeq[String](0)
          }
          loggingErrorStats("NULL error detector", qualifiedName, errCellView, attrsToRepair)
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

      withTempView(inputDf, cache = true) { inputView =>
        val constraints = loadConstraintsFromFile(constraintFilePath, tableName, inputDf.columns)
        if (constraints.entries.isEmpty) {
          // Case of non-found constraints
          val rowIdType = inputDf.schema.find(_.name == rowId).get.dataType.sql
          createEmptyTable(s"$rowId $rowIdType, attribute STRING")
        } else {
          logBasedOnLevel({
            val constraintLists = constraints.entries.zipWithIndex.map { case (preds, i) =>
              preds.map(_.toString("t1", "t2")).mkString(s" [$i] ", ",", "")
            }
            s"""
               |Loads constraints from '$constraintFilePath':
               |${constraintLists.mkString("\n")}
             """.stripMargin
          })

          // Detects error erroneous cells in a given table
          val errCellDf = constraints.entries.flatMap { preds =>
            val queryToValidateConstraint =
              s"""
                 |SELECT t1.$rowId
                 |FROM $inputView AS t1
                 |WHERE EXISTS (
                 |  SELECT t2.$rowId
                 |  FROM $inputView AS t2
                 |  WHERE ${DenialConstraints.toWhereCondition(preds, "t1", "t2")}
                 |)
               """.stripMargin

            val df = sparkSession.sql(queryToValidateConstraint)
            logBasedOnLevel(
              s"""
                 |Number of violate tuples: ${df.count}
                 |Query to validate constraints:
                 |$queryToValidateConstraint
               """.stripMargin)

            preds.flatMap { p => p.leftAttr :: p.rightAttr :: Nil }.map { attr =>
              df.selectExpr(rowId, s"'$attr' AS attribute")
            }
          }.reduce(_.union(_)).distinct().cache()

          val errCellView = createAndCacheTempView(errCellDf, "err_cells_from_constraints")
          loggingErrorStats("Constraint error detector", qualifiedName, errCellView, constraints.attrNames)
          errCellDf
        }
      }
    }
  }
}
