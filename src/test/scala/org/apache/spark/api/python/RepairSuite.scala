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

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.sql.QueryTest
import org.apache.spark.SparkException
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

class RepairSuite extends QueryTest with SharedSparkSession {

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sql(s"SET ${SQLConf.CBO_ENABLED.key}=true")
  }

  test("checkInputTable - unsupported types") {
    Seq("DATE", "TIMESTAMP", "ARRAY<INT>", "STRUCT<a: INT, b: DOUBLE>", "MAP<INT, INT>")
        .foreach { tpe =>
      withTable("t") {
        spark.sql(s"CREATE TABLE t(tid STRING, v $tpe) USING parquet")
        val errMsg = intercept[SparkException] {
          RepairApi.checkInputTable("default", "t", "tid")
        }.getMessage
        assert(errMsg.contains("unsupported ones found"))
      }
    }
  }

  test("checkInputTable") {
    withTempView("t") {
      spark.range(1).selectExpr("CAST(id AS STRING) tid", "id AS v1", "CAST(id AS DOUBLE) v2")
        .createOrReplaceTempView("t")
      val jsonString = RepairApi.checkInputTable("", "t", "tid")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data("input_table") === "t")
      assert(data("num_input_rows") === "1")
      assert(data("num_attrs") === "3")
      assert(data("continous_attrs") === "v2")
    }
  }

  test("computeAndGetTableStats") {
    withTempView("t") {
      spark.range(30).selectExpr(
        "CAST(id % 2 AS BOOLEAN) AS v0",
        "CAST(id % 3 AS LONG) AS v1",
        "CAST(id % 8 AS DOUBLE) AS v2",
        "CAST(id % 6 AS STRING) AS v3"
      ).createOrReplaceTempView("t")
      val statMap = RepairApi.computeAndGetTableStats("t")
      assert(statMap.mapValues(_.distinctCount) ===
        Map("v0" -> 2, "v1" -> 3, "v2" -> 8, "v3" -> 6))
    }
  }

  test("computeDomainSizes") {
    withTempView("t") {
      spark.range(30).selectExpr("id % 3 AS v0", "id % 8 AS v1", "id % 6 AS v2", "id % 9 AS v3")
        .createOrReplaceTempView("t")
      val jsonString = RepairApi.computeDomainSizes("t")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data("distinct_stats") === Map("v0" -> 3, "v1" -> 8, "v2" -> 6, "v3" -> 9))
    }
  }
}
