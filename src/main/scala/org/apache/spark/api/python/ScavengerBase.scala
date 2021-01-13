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
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.util.LoggingBasedOnLevel
import org.apache.spark.util.ScavengerUtils._
import org.apache.spark.sql.types._

private[python] case class JsonEncoder(v: Seq[(String, AnyRef)]) {

  def asJson: String = v.map {
    case (k, v: String) => s""""$k":"$v""""
    case (k, map: Map[AnyRef, AnyRef]) =>
      // TODO: We cannot wrap values with `"` because of non-string cases
      s""""$k":${map.map(kv => s""""${kv._1}": ${kv._2}""").mkString("{", ",", "}")}"""
  }.mkString("{", ",", "}")
}

abstract class ScavengerBase extends LoggingBasedOnLevel {

  protected val continousTypes: Set[DataType] = Set(FloatType, DoubleType)
  protected val supportedType: Set[DataType] = Set(StringType, BooleanType, ByteType, ShortType,
    IntegerType, LongType) ++ continousTypes

  protected def spark = {
    assert(SparkSession.getActiveSession.nonEmpty, "active Spark session not found")
    SparkSession.getActiveSession.get
  }

  protected implicit def seqToJsonEncoder(ar: Seq[(String, AnyRef)]) = JsonEncoder(ar)

  // TODO: We need a smarter way to convert Scala data to a json string
  protected def seqToJson(seq: Seq[(Any, Any)]): String = {
    seq.map(v => s"""["${v._1}","${v._2}"]""").mkString("[", ",", "]")
  }

  protected def withSQLConf[T](pairs: (String, String)*)(f: => T): T= {
    val conf = SQLConf.get
    val (keys, values) = pairs.unzip
    val currentValues = keys.map { key =>
      if (conf.contains(key)) {
        Some(conf.getConfString(key))
      } else {
        None
      }
    }
    (keys, values).zipped.foreach { (k, v) =>
      assert(!SQLConf.staticConfKeys.contains(k))
      conf.setConfString(k, v)
    }
    try f finally {
      keys.zip(currentValues).foreach {
        case (key, Some(value)) => conf.setConfString(key, value)
        case (key, None) => conf.unsetConf(key)
      }
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

  protected def checkIfColumnsExistIn(tableName: String, expectedColumns: Seq[String]): Unit = {
    val columnsInRepairedCells = spark.table(tableName).columns
    if (!expectedColumns.forall(columnsInRepairedCells.contains)) {
      throw new SparkException(s"'$tableName' must have " +
        s"${expectedColumns.map(c => s"'$c'").mkString(", ")} columns.")
    }
  }

  def checkAndGetInputTable(dbName: String, tableName: String, rowId: String = "")
    : (DataFrame, String) = {
    val inputName = if (dbName.nonEmpty) s"$dbName.$tableName" else tableName
    val inputDf = spark.table(inputName)
    // Checks if the given table has a column named `rowId`
    if (rowId.nonEmpty && !inputDf.columns.contains(rowId)) {
      // TODO: Implicitly adds unique row IDs if they don't exist in a given table
      throw new SparkException(s"Column '$rowId' does not exist in '$inputName'.")
    }
    if (inputDf.columns.length <= 1) {
      throw new SparkException(
        s"At least one valid column needs to exist, but only one column '$rowId' exists.")
    }
    (inputDf, inputName)
  }
}
