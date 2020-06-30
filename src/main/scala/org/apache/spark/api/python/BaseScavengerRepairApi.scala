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

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.python._
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

private[python] case class JsonEncoder(v: Seq[(String, AnyRef)]) {

  def asJson: String = v.map {
    case (k, v: String) => s""""$k":"$v""""
    case (k, map: Map[AnyRef, AnyRef]) =>
      // TODO: We cannot wrap values with `"` because of non-string cases
      s""""$k":${map.map(kv => s""""${kv._1}": ${kv._2}""").mkString("{", ",", "}")}"""
  }.mkString("{", ",", "}")
}

class BaseScavengerRepairApi extends Logging {

  protected implicit def seqToJsonEncoder(ar: Seq[(String, AnyRef)]) = JsonEncoder(ar)

  // TODO: We need a smarter way to convert Scala data to a json string
  protected def seqToJson(seq: Seq[(Any, Any)]): String = {
    seq.map(v => s"""["${v._1}","${v._2}"]""").mkString("[", ",", "]")
  }

  protected def logBasedOnLevel(msg: => String): Unit = {
    // This method should be called inside `withSparkSession`
    val spark = SparkSession.getActiveSession.get
    spark.sessionState.conf.logLevel match {
      case "TRACE" => logTrace(msg)
      case "DEBUG" => logDebug(msg)
      case "INFO" => logInfo(msg)
      case "WARN" => logWarning(msg)
      case "ERROR" => logError(msg)
      case _ => logTrace(msg)
    }
  }

  protected def checkAndGetInputTable(dbName: String, tableName: String, rowId: String = "")
    : (DataFrame, String) = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val inputName = if (dbName.nonEmpty) s"$dbName.$tableName" else tableName
    val inputDf = spark.table(inputName)
    // Checks if the given table has a column named `rowId`
    if (rowId.nonEmpty && !inputDf.columns.contains(rowId)) {
      // TODO: Implicitly adds unique row IDs if they don't exist in a given table
      throw new SparkException(s"Column '$rowId' does not exist in $inputName.")
    }
    (inputDf, inputName)
  }

  protected def checkIfColumnsExistIn(tableName: String, expectedColumns: Seq[String]): Unit = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val columnsInRepairedCells = spark.table(tableName).columns
    if (!expectedColumns.forall(columnsInRepairedCells.contains)) {
      throw new SparkException(s"$tableName must have " +
        s"${expectedColumns.map(c => s"'$c'").mkString(", ")} columns.")
    }
  }

  protected def loadConstraintsFromFile(constraintFilePath: String, inputName: String, tableAttrs: Seq[String]): DenialConstraints = {
    // Loads all the denial constraints from a given file path
    val allConstraints = DenialConstraints.parse(constraintFilePath)
    // Checks if all the attributes contained in `constraintFilePath` exist in `table`
    val attrsInConstraints = allConstraints.attrNames
    val tableAttrSet = tableAttrs.toSet
    val absentAttrs = attrsInConstraints.filterNot(tableAttrSet.contains)
    if (absentAttrs.nonEmpty) {
      logWarning(s"Non-existent constraint attributes found in $inputName: ${absentAttrs.mkString(", ")}")
      val newPredEntries = allConstraints.entries.filter { _.forall { p =>
        tableAttrSet.contains(p.leftAttr) && tableAttrSet.contains(p.rightAttr)
      }}
      if (newPredEntries.nonEmpty) {
        allConstraints.copy(entries = newPredEntries)
      } else {
        DenialConstraints.emptyConstraints
      }
    } else {
      allConstraints
    }
  }

  protected def createAndCacheTempView(df: DataFrame, name: String = ""): String = {
    def timer(name: String)(computeRowCnt: => Long): Unit = {
      val t0 = System.nanoTime()
      val rowCnt = computeRowCnt
      val t1 = System.nanoTime()
      logBasedOnLevel(s"Elapsed time to compute '$name' with $rowCnt rows: " +
        ((t1 - t0 + 0.0) / 1000000000.0) + "s")
    }
    val tempViewId = getRandomString()
    val numShufflePartitions = df.sparkSession.sessionState.conf.numShufflePartitions
    df.coalesce(numShufflePartitions).cache.createOrReplaceTempView(tempViewId)
    timer(if (name.nonEmpty) s"$name($tempViewId)" else tempViewId) {
      df.sparkSession.table(tempViewId).count()
    }
    tempViewId
  }

  protected def createEmptyTable(schema: String): DataFrame = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    spark.createDataFrame(spark.sparkContext.emptyRDD[Row], StructType.fromDDL(schema))
  }
}
