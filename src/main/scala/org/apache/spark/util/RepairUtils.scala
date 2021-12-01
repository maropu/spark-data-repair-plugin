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

package org.apache.spark.util

import org.apache.commons.lang.RandomStringUtils

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object RepairUtils {

  def withJobDescription[T](desc: String)(f: => T): T = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val currentJobDesc = spark.sparkContext.getLocalProperty("spark.job.description")
    spark.sparkContext.setJobDescription(desc)
    val ret = f
    spark.sparkContext.setJobDescription(currentJobDesc)
    ret
  }

  def withTempView[T](df: DataFrame, prefix: String, cache: Boolean = false)(f: String => T): T = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val tempView = getRandomString(s"${prefix}_")
    if (cache) df.cache()
    df.createOrReplaceTempView(tempView)
    val ret = f(tempView)
    SparkSession.getActiveSession.get.sql(s"DROP VIEW $tempView")
    ret
  }

  def checkSchema(
      df: DataFrame,
      expectedSchemDDL: String,
      rowId: String,
      strict: Boolean): Boolean = {
    val viewSchema = df.schema
    val hasRowId = viewSchema.exists(_.name == rowId)
    val expectedSchema = StructType.fromDDL(expectedSchemDDL)
    val hasEnoughFields = !strict || viewSchema.length - 1 >= expectedSchema.length
    val viewFieldSet = viewSchema.map(f => (f.name, f.dataType)).toSet
    hasRowId && hasEnoughFields && expectedSchema.forall { f =>
      viewFieldSet.contains((f.name, f.dataType))
    }
  }

  def checkSchema(
      viewName: String,
      expectedSchemDDL: String,
      rowId: String,
      strict: Boolean): Boolean = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    checkSchema(spark.table(viewName), expectedSchemDDL, rowId, strict)
  }

  def createEmptyTable(schema: String): DataFrame = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    spark.createDataFrame(spark.sparkContext.emptyRDD[Row], StructType.fromDDL(schema))
  }

  def getRandomString(prefix: String = ""): String = {
    val prefixStr = if (prefix.nonEmpty) prefix else Utils.getFormattedClassName(this)
    s"${prefixStr}_${RandomStringUtils.randomNumeric(16)}"
  }
}
