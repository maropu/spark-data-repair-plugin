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
    Seq("BOOLEAN", "DATE", "TIMESTAMP", "ARRAY<INT>", "STRUCT<a: INT, b: DOUBLE>", "MAP<INT, INT>")
        .foreach { tpe =>
      withTable("t") {
        spark.sql(s"CREATE TABLE t(tid STRING, v1 STRING, v2 $tpe) USING parquet")
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
      val supportedTypes = Seq("BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "STRING")
      val exprs = "CAST(id AS INT) tid" +:
        supportedTypes.zipWithIndex.map { case (t, i) => s"CAST(id AS $t) AS v$i" }
      spark.range(1).selectExpr(exprs: _*).createOrReplaceTempView("t")
      val jsonString = RepairApi.checkInputTable("", "t", "tid")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data("input_table") === "t")
      assert(data("num_input_rows") === "1")
      assert(data("num_attrs") === "7")
      assert(data("continous_attrs") === "v0,v1,v2,v3,v4,v5")
    }
  }

  test("withCurrentValues") {
    import testImplicits._
    withTempView("inputView", "errCell") {
      Seq(
        (1, 100, "abc", 1.2),
        (2, 200, "def", 3.2),
        (3, 300, "ghi", 2.1),
        (4, 400, "jkl", 1.9),
        (5, 500, "mno", 0.5)
      ).toDF("tid", "c0", "c1", "c2").createOrReplaceTempView("inputView")

      Seq((2, "c1"), (2, "c2"), (3, "c0"), (5, "c2"))
        .toDF("tid", "attribute").createOrReplaceTempView("errCell")

      val df = RepairApi.withCurrentValues("inputView", "errCell", "tid", "c0,c1,c2")
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

  test("convertErrorCellsToNull") {
    import testImplicits._
    withTempView("inputView", "errCell") {
      Seq(
        (1, 100, "abc", 1.2),
        (2, 200, "def", 3.2),
        (3, 300, "ghi", 2.1),
        (4, 400, "jkl", 1.9),
        (5, 500, "mno", 0.5)
      ).toDF("tid", "c0", "c1", "c2").createOrReplaceTempView("inputView")

      Seq((2, "c1", "def"), (2, "c2", "3.2"), (3, "c0", "300"), (5, "c2", "0.5"))
        .toDF("tid", "attribute", "current_value").createOrReplaceTempView("errCell")

      val jsonString = RepairApi.convertErrorCellsToNull("inputView", "errCell", "tid", "c0,c1,c2")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values

      val viewName = data("repair_base_cells").toString
      assert(viewName.startsWith("repair_base_cells_"))
      checkAnswer(spark.table(viewName), Seq(
        Row(1, 100, "abc", 1.2),
        Row(2, 200, null, null),
        Row(3, null, "ghi", 2.1),
        Row(4, 400, "jkl", 1.9),
        Row(5, 500, "mno", null)
      ))
    }
  }

  test("computeAttrStats") {
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

      val df = RepairApi.computeAttrStats(
        "tempView", Seq("x", "y"), Seq(("x", "y"), ("y", "x")), 1.0, 0.0)
      checkAnswer(df, Seq(
        Row("1", "test-1", 3),
        Row("2", "test-2a", 1),
        Row(null, "test-2a", 1),
        Row("2", "test-2", 2),
        Row(null, "test-2", 2),
        Row("3", null, 3),
        Row("3", "test-3", 3),
        Row(null, "test-1", 3),
        Row("2", null, 3),
        Row(null, "test-3", 3),
        Row("1", null, 3)
      ))
    }
  }

  test("compuatePairwiseAttrStats") {
    withTempView("tempView", "attrStatView") {
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

      val statThreshold = 0.80
      val df = RepairApi.computeAttrStats(
        "tempView", Seq("x", "y"), Seq(("x", "y"), ("y", "x")), 1.0, statThreshold)
      df.createOrReplaceTempView("attrStatView")

      val (pairwiseStatMap, domainStatMap) = RepairApi.compuatePairwiseAttrStats(
        9, "attrStatView", "tempView", Seq("x", "y"), Seq(("x", "y"), ("y", "x")), 0.3)
      assert(pairwiseStatMap.keySet === Set("x", "y"))
      assert(pairwiseStatMap("x").map(_._1) === Seq("y"))
      assert(pairwiseStatMap("x").head._2 > statThreshold)
      assert(pairwiseStatMap("y").map(_._1) === Seq("x"))
      assert(pairwiseStatMap("y").head._2 > statThreshold)
      assert(domainStatMap === Map("tid" -> 9, "x" -> 3, "y" -> 4))
    }
  }

  test("computeCorrAttrs") {
    val pairwiseStatMap = Map("y" -> Seq(("x", 0.9)), "x" -> Seq(("y", 0.9)))
    val corrAttrs = RepairApi.computeCorrAttrs(pairwiseStatMap, 2, 0.80)
    assert(corrAttrs === Map("y" -> Seq(("x", 0.9)), "x" -> Seq(("y", 0.9))))
  }

  test("computeDomainInErrorCells") {
    withTempView("inputView", "errCell") {
      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW inputView(tid, x, y, z) AS SELECT * FROM VALUES
           |  (1, "2", "test-1", 1.0),
           |  (2, "2", "test-2", 1.0),
           |  (3, "3", "test-1", 3.0),
           |  (4, "2", "test-2", 2.0),
           |  (5, "1", "test-1", 1.0),
           |  (6, "2", "test-1", 1.0),
           |  (7, "3", "test-3", 2.0),
           |  (8, "3", "test-3", 3.0),
           |  (9, "2", "test-2a", 2.0)
         """.stripMargin)

      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW errCell(tid, attribute, current_value) AS SELECT * FROM VALUES
           |  (1, "x", "2"),
           |  (3, "y", "test-3"),
           |  (6, "y", "test-2")
         """.stripMargin)

      def testComputeDomain(minCorrThres: Double, domain_threshold_beta: Double, expected: Seq[Row]): Unit = {
        val jsonString = RepairApi.computeDomainInErrorCells(
          "inputView", "errCell", "tid", "z", "x,y", "x,y", 9, 4, 1.0, 0.0, minCorrThres, 0.0, domain_threshold_beta)
        val jsonObj = parse(jsonString)
        val data = jsonObj.asInstanceOf[JObject].values

        val viewName = data("cell_domain").toString
        assert(viewName.startsWith("cell_domain_"))
        val domainDf = spark.table(viewName)
        assert(domainDf.columns.toSet === Set("tid", "attribute", "current_value", "domain"))
        val df = spark.table(viewName)
          .selectExpr("*", "inline(domain)")
          .selectExpr("tid", "attribute", "current_value", "n")
        checkAnswer(df, expected)
      }

      testComputeDomain(0.0, 0.0, Seq(
        Row(1, "x", "2", "1"),
        Row(1, "x", "2", "2"),
        Row(1, "x", "2", "3"),
        Row(3, "y", "test-3", "test-1"),
        Row(3, "y", "test-3", "test-3"),
        Row(6, "y", "test-2", "test-1"),
        Row(6, "y", "test-2", "test-2"),
        Row(6, "y", "test-2", "test-2a")
      ))
    }
  }
}
