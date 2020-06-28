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
import org.apache.spark.util.{Utils => SparkUtils}

object ScavengerMiscApi extends BaseScavengerRepairApi {

  def injectNullAt(dbName: String, tableName: String, targetAttrList: String, nullRatio: Double): String = {
    logBasedOnLevel(s"injectNullAt called with: dbName=$dbName tableName=$tableName " +
      s"targetAttrList=$targetAttrList, nullRatio=$nullRatio")

    val df = withSparkSession { _ =>
      val (inputDf, qualifiedName) = checkAndGetInputTable(dbName, tableName)
      val targetAttrSet = if (targetAttrList.nonEmpty) {
        val attrSet = SparkUtils.stringToSeq(targetAttrList).toSet
        if (!inputDf.columns.exists(attrSet.contains)) {
          throw new SparkException(s"No target attribute selected in $qualifiedName")
        }
        attrSet
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
    Seq("injected" -> createAndCacheTempView(df)).asJson
  }

  /**
   * To compare result rows easily, this method flattens an input table
   * as a schema (`rowId`, attribute, val).
   */
  def flattenTable(dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"flattenTable called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    val df = withSparkSession { _ =>
      val (inputDf, _) = checkAndGetInputTable(dbName, tableName, rowId)
      val expr = inputDf.schema.filter(_.name != rowId)
        .map { f => s"STRUCT($rowId, '${f.name}', CAST(${f.name} AS STRING))" }
        .mkString("ARRAY(", ", ", ")")
      inputDf.selectExpr(s"INLINE($expr) AS (tid, attribute, value)")
    }
    Seq("flatten" -> createAndCacheTempView(df)).asJson
  }
}
