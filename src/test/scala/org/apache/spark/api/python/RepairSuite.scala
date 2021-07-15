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
import org.apache.spark.sql.{AnalysisException, QueryTest, Row}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

class RepairSuite extends QueryTest with SharedSparkSession {

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sql(s"SET ${SQLConf.CBO_ENABLED.key}=true")
  }

  private def resourcePath(f: String): String = {
    Thread.currentThread().getContextClassLoader.getResource(f).getPath
  }

  test("checkInputTable - type check") {
    Seq("DATE", "TIMESTAMP", "ARRAY<INT>", "STRUCT<a: INT, b: DOUBLE>", "MAP<INT, INT>")
        .foreach { tpe =>
      withTable("t") {
        spark.sql(s"CREATE TABLE t(tid STRING, v $tpe) USING parquet")
        val errMsg = intercept[AnalysisException] {
          RepairApi.checkInputTable("default", "t", "tid")
        }.getMessage
        assert(errMsg.contains("unsupported ones found"))
      }
    }
  }

  test("checkInputTable - #columns check") {
    Seq("tid STRING", "tid STRING, c1 INT").foreach { schema =>
      withTable("t") {
        spark.sql(s"CREATE TABLE t($schema) USING parquet")
        val errMsg = intercept[AnalysisException] {
          RepairApi.checkInputTable("default", "t", "tid")
        }.getMessage
        assert(errMsg.contains("A least three columns"))
      }
    }
  }

  test("checkInputTable - uniqueness check") {
    withTempView("t") {
      spark.range(100).selectExpr("1 AS tid", "id % 2 AS c0", " id % 3 AS c0")
        .createOrReplaceTempView("t")
      val errMsg = intercept[AnalysisException] {
        RepairApi.checkInputTable("", "t", "tid")
      }.getMessage
      assert(errMsg.contains("Uniqueness does not hold"))
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
      assert(data("num_attrs") === "2")
      assert(data("continous_attrs") === "v2")
    }
  }

  test("withCurrentValues") {
    import testImplicits._
    withTempView("input", "err") {
      Seq(
        (1, 100, "abc", 1.2),
        (2, 200, "def", 3.2),
        (3, 300, "ghi", 2.1),
        (4, 400, "jkl", 1.9),
        (5, 500, "mno", 0.5)
      ).toDF("tid", "c0", "c1", "c2").createOrReplaceTempView("input")

      Seq((2, "c1"), (2, "c2"), (3, "c0"), (5, "c2"))
        .toDF("tid", "attribute").createOrReplaceTempView("err")

      val df = RepairApi.withCurrentValues("input", "err", "tid", "c0,c1,c2")
      checkAnswer(df, Seq(
        Row(2, "c1", "def"),
        Row(2, "c2", "3.2"),
        Row(3, "c0", "300"),
        Row(5, "c2", "0.5")
      ))
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

  test("computeFunctionalDeps") {
    withTempView("hospital") {
      val hospitalFilePath = resourcePath("hospital.csv")
      spark.read.option("header", true).format("csv").load(hospitalFilePath).createOrReplaceTempView("hospital")
      val constraintFilePath = resourcePath("hospital_constraints.txt")
      val jsonString = RepairApi.computeFunctionalDeps("hospital", constraintFilePath)
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values

      assert(data === Map(
        "HospitalOwner" -> Seq("HospitalName"),
        "Condition" -> Seq("MeasureCode"),
        "CountyName" -> Seq("City"),
        "HospitalName" -> Seq("Address1", "City", "PhoneNumber", "ProviderNumber"),
        "EmergencyService" -> Seq("ZipCode"),
        "ZipCode" -> Seq("HospitalName"),
        "MeasureCode" -> Seq("MeasureName", "Stateavg")))
    }
  }

  test("computeFunctionalDepMap") {
    withTempView("tempView") {
      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW tempView(tid, x, y) AS SELECT * FROM VALUES
           |  (1, "1", "test-1"),
           |  (2, "2", "test-2"),
           |  (3, "3", "test-3"),
           |  (4, "2", "test-2"),
           |  (5, "1", "test-1"),
           |  (6, "1", "test-1"),
           |  (7, "3", "test-3"),
           |  (8, "3", "test-3"),
           |  (9, "2", "test-2a")
         """.stripMargin)

      val jsonString = RepairApi.computeFunctionDepMap("tempView", "x", "y")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data === Map("3" -> "test-3", "1" -> "test-1"))
    }
  }

  test("convertToDiscretizedTable") {
    withTable("adult") {
      val hospitalFilePath = resourcePath("hospital.csv")
      spark.read.option("header", true).format("csv").load(hospitalFilePath).write.saveAsTable("hospital")
      val jsonString = RepairApi.convertToDiscretizedTable("default.hospital", "tid", 20)
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values

      val discretizedTable = data("discretized_table").toString
      assert(discretizedTable.startsWith("discretized_table_"))
      val discretizedCols = spark.table(discretizedTable).columns
      assert(discretizedCols.toSet === Set("tid", "HospitalType", "EmergencyService", "State"))

      assert(data("distinct_stats") === Map(
        "HospitalOwner" -> 28,
        "MeasureName" -> 63,
        "Address2" -> 0,
        "Condition" -> 28,
        "Address3" -> 0,
        "PhoneNumber" -> 72,
        "CountyName" -> 65,
        "ProviderNumber" -> 71,
        "HospitalName" -> 68,
        "Sample" -> 355,
        "HospitalType" -> 13,
        "EmergencyService" -> 6,
        "City" -> 72,
        "Score" -> 71,
        "ZipCode" -> 67,
        "Address1" -> 78,
        "State" -> 4,
        "tid" -> 1000,
        "Stateavg" -> 74,
        "MeasureCode" -> 56))
    }
  }
}
