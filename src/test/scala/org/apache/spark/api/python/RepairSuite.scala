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
      assert(data("continous_attrs") === "v0,v1,v2,v3,v4,v5")
    }

  }

  test("withCurrentValues") {
    import testImplicits._
    Seq(("tid", "c0", "c1", "c2"), ("t i d", "c 0", "c 1", "c 2")).foreach { case (tid, c0, c1, c2) =>
      withTempView("inputView", "errCell") {
        Seq(
          (1, 100, "abc", 1.2),
          (2, 200, "def", 3.2),
          (3, 300, "ghi", 2.1),
          (4, 400, "jkl", 1.9),
          (5, 500, "mno", 0.5)
        ).toDF(tid, c0, c1, c2).createOrReplaceTempView("inputView")

        Seq((2, c1), (2, c2), (3, c0), (5, c2))
          .toDF(tid, "attribute").createOrReplaceTempView("errCell")

        val df = RepairApi.withCurrentValues("inputView", "errCell", tid, s"$c0,$c1,$c2")
        checkAnswer(df, Seq(
          Row(2, s"$c1", "def"),
          Row(2, s"$c2", "3.2"),
          Row(3, s"$c0", "300"),
          Row(5, s"$c2", "0.5")
        ))
      }
    }
  }

  test("computeAndGetTableStats") {
    Seq(("v0", "v1", "v2", "v3"), ("v 0", "v 1", "v 2", "v 3")).foreach { case (v0, v1, v2, v3) =>
      withTempView("t") {
        spark.range(30).selectExpr(
          s"CAST(id % 2 AS BOOLEAN) AS `$v0`",
          s"CAST(id % 3 AS LONG) AS `$v1`",
          s"CAST(id % 8 AS DOUBLE) AS `$v2`",
          s"CAST(id % 6 AS STRING) AS `$v3`"
        ).createOrReplaceTempView("t")
        val statMap = RepairApi.computeAndGetTableStats("t")
        assert(statMap.mapValues(_.distinctCount) ===
          Map(s"$v0" -> 2, s"$v1" -> 3, s"$v2" -> 8, s"$v3" -> 6))
      }
    }
  }

  test("computeDomainSizes") {
    Seq(("v0", "v1", "v2", "v3"), ("v 0", "v 1", "v 2", "v 3")).foreach { case (v0, v1, v2, v3) =>
      withTempView("t") {
        spark.range(30).selectExpr(s"id % 3 AS `$v0`", s"id % 8 AS `$v1`", s"id % 6 AS `$v2`", s"id % 9 AS `$v3`")
          .createOrReplaceTempView("t")
        val jsonString = RepairApi.computeDomainSizes("t")
        val jsonObj = parse(jsonString)
        val data = jsonObj.asInstanceOf[JObject].values
        assert(data("domain_stats") === Map(s"$v0" -> 3, s"$v1" -> 8, s"$v2" -> 6, s"$v3" -> 9))
      }
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
      assert(data("domain_stats") === Map(
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
        "Stateavg" -> 74,
        "MeasureCode" -> 56))
    }
  }

  test("convertToDiscretizedTable - escaped column names") {
    import testImplicits._
    withTempView("inputView", "errCell") {
      Seq(
        (1, 100, "abc", 1.2),
        (2, 200, "def", 3.2),
        (3, 100, "def", 2.1),
        (4, 100, "abc", 1.9),
        (5, 200, "abc", 0.5)
      ).toDF("t i d", "c 0", "c 1", "c 2").createOrReplaceTempView("inputView")

      val jsonString = RepairApi.convertToDiscretizedTable("inputView", "t i d", 2)
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values

      val discretizedTable = data("discretized_table").toString
      assert(discretizedTable.startsWith("discretized_table_"))
      val discretizedCols = spark.table(discretizedTable).columns
      assert(discretizedCols.toSet === Set("t i d", "c 0", "c 1", "c 2"))
      assert(data("domain_stats") === Map(
        "c 0" -> 2,
        "c 1" -> 2,
        "c 2" -> 5))
    }
  }

  test("convertErrorCellsToNull") {
    import testImplicits._
    Seq(("tid", "c0", "c1", "c2"), ("t i d", "c 0", "c 1", "c 2")).foreach { case (tid, c0, c1, c2) =>
      withTempView("inputView", "errCell") {
        Seq(
          (1, 100, "abc", 1.2),
          (2, 200, "def", 3.2),
          (3, 300, "ghi", 2.1),
          (4, 400, "jkl", 1.9),
          (5, 500, "mno", 0.5)
        ).toDF(tid, c0, c1, c2).createOrReplaceTempView("inputView")

        Seq((2, c1, "def"), (2, c2, "3.2"), (3, c0, "300"), (5, c2, "0.5"))
          .toDF(tid, "attribute", "current_value").createOrReplaceTempView("errCell")

        val jsonString = RepairApi.convertErrorCellsToNull("inputView", "errCell", tid, s"$c0,$c1,$c2")
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
  }

  test("computeFreqStats") {
    Seq(("tid", "xx", "yy"), ("t i d", "x x", "y y")).foreach { case (tid, x, y) =>
      withTempView("tempView") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW tempView(`$tid`, `$x`, `$y`) AS SELECT * FROM VALUES
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

        val attrsToComputeFreqStats = Seq(Seq(x), Seq(y), Seq(x, y))
        val df1 = RepairApi.computeFreqStats("tempView", attrsToComputeFreqStats, 1.0, 0.0)
        checkAnswer(df1, Seq(
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
        val df2 = RepairApi.computeFreqStats("tempView", attrsToComputeFreqStats, 1.0, 0.3)
        checkAnswer(df2, Seq(
          Row("1", "test-1", 3),
          Row("3", null, 3),
          Row("3", "test-3", 3),
          Row(null, "test-1", 3),
          Row("2", null, 3),
          Row(null, "test-3", 3),
          Row("1", null, 3)
        ))

        val errMsg = intercept[IllegalStateException] {
          RepairApi.computeFreqStats("tempView", Seq(Seq(tid, x, y)), 1.0, 0.0)
        }.getMessage
        assert(errMsg.contains(s"Cannot handle more than two entries: $tid,$x,$y"))
      }
    }
  }

  test("computePairwiseStats") {
    Seq(("tid", "xx", "yy"), ("t i d", "x x", "y y")).foreach { case (tid, x, y) =>
      withTempView("tempView", "freqAttrStats") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW tempView(`$tid`, `$x`, `$y`) AS SELECT * FROM VALUES
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

        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW freqAttrStats(`$x`, `$y`, cnt) AS SELECT * FROM VALUES
             |  ("2", "test-2", 2),
             |  ("2", "test-1", 2),
             |  ("3", "test-1", 1),
             |  ("1", "test-1", 1),
             |  ("2", "test-2a", 1),
             |  ("3", "test-3", 2),
             |  (null, "test-2", 2),
             |  ("3", null, 3),
             |  ("2", null, 5),
             |  (null, "test-1", 4),
             |  (null, "test-2a", 1),
             |  (null, "test-3", 2),
             |  ("1", null, 1)
             """.stripMargin)

        val domainStatMap = Map(tid -> 9L, s"$x" -> 3L, s"$y" -> 4L)
        val pairwiseStatMap = RepairApi.computePairwiseStats(
          "tempView", 9, "freqAttrStats", Seq(s"$x", s"$y"), Seq((s"$x", s"$y"), (s"$y", s"$x")), domainStatMap)
        assert(pairwiseStatMap.keySet === Set(s"$x", s"$y"))
        assert(pairwiseStatMap(s"$x").map(_._1) === Seq(s"$y"))
        assert(pairwiseStatMap(s"$x").head._2 > 0.0)
        assert(pairwiseStatMap(s"$y").map(_._1) === Seq(s"$x"))
        assert(pairwiseStatMap(s"$y").head._2 > 0.0)
      }
    }
  }

  test("computeAttrStats") {
    Seq(("tid", "xx", "yy"), ("t i d", "x x", "y y")).foreach { case (tid, x, y) =>
      withTempView("tempView", "attrStatView") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW tempView(`$tid`, `$x`, `$y`) AS SELECT * FROM VALUES
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

        val statThreshold = 0.0
        val domainStatMapAsJson = s"""{"$tid": 9,"$x": 3,"$y": 4}"""
        val jsonString = RepairApi.computeAttrStats(
          "tempView", tid, s"$x,$y", domainStatMapAsJson, 1.0, statThreshold)

        val jsonObj = parse(jsonString)
        val data = jsonObj.asInstanceOf[JObject].values

        val freqAttrStatView = data("freq_attr_stats").toString
        checkAnswer(spark.table(freqAttrStatView), Seq(
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

        val pairwiseStatMap = data("pairwise_attr_stats")
          .asInstanceOf[Map[String, Seq[Seq[String]]]]
          .mapValues(_.map { case Seq(attr, sv) => (attr, sv.toDouble) })
        assert(pairwiseStatMap.keySet === Set(s"$x", s"$y"))
        assert(pairwiseStatMap(s"$x").head._1 === s"$y")
        assert(pairwiseStatMap(s"$x").head._2 > statThreshold)
        assert(pairwiseStatMap(s"$y").head._1 === s"$x")
        assert(pairwiseStatMap(s"$y").head._2 > statThreshold)
      }
    }
  }

  test("computeCorrAttrs") {
    val pairwiseStatMap = Map("y" -> Seq(("x", 0.9)), "x" -> Seq(("y", 0.9)))
    val corrAttrs = RepairApi.filterCorrAttrs(pairwiseStatMap, 2, 0.80)
    assert(corrAttrs === Map("y" -> Seq(("x", 0.9)), "x" -> Seq(("y", 0.9))))
  }

  test("computeDomainInErrorCells") {
    Seq(("tid", "xx", "yy", "zz"), ("t i d", "x x", "y y", "z z")).foreach { case (tid, x, y, z) =>
      withTempView("inputView", "errCell", "freqAttrStats") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW inputView(`$tid`, `$x`, `$y`, `$z`) AS SELECT * FROM VALUES
             |  (1, "2", "test-1", 1),
             |  (2, "2", "test-2", 1),
             |  (3, "3", "test-1", 3),
             |  (4, "2", "test-2", 2),
             |  (5, "1", "test-1", 1),
             |  (6, "2", "test-1", 1),
             |  (7, "3", "test-3", 2),
             |  (8, "3", "test-3", 3),
             |  (9, "2", "test-2a", 2)
           """.stripMargin)

        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW errCell(`$tid`, attribute, current_value) AS SELECT * FROM VALUES
             |  (1, "$x", "2"),
             |  (3, "$y", "test-3"),
             |  (6, "$y", "test-2")
           """.stripMargin)

        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW freqAttrStats(`$x`, `$y`, `$z`, cnt) AS SELECT * FROM VALUES
             |  ("2", null, 1, 3),
             |  (null, "test-1", 1, 3),
             |  ("2", null, 2, 2),
             |  ("2", null, null, 5),
             |  (null, "test-1", null, 4),
             |  ("2", "test-2", null, 2),
             |  (null, null, 3, 2),
             |  (null, "test-1", 3, 1),
             |  (null, "test-2", 1, 1),
             |  ("3", null, null, 3),
             |  (null, "test-2", 2, 1),
             |  (null, null, 1, 4),
             |  ("2", "test-1", null, 2),
             |  (null, "test-2", null, 2),
             |  ("3", "test-1", null, 1),
             |  (null, null, 2, 3),
             |  ("3", null, 3, 2),
             |  ("1", "test-1", null, 1),
             |  (null, "test-3", 2, 1),
             |  (null, "test-3", 3, 1),
             |  (null, "test-2a", 2, 1),
             |  ("1", null, null, 1),
             |  ("3", "test-3", null, 2),
             |  (null, "test-2a", null, 1),
             |  ("2", "test-2a", null, 1),
             |  (null, "test-3", null, 2),
             |  ("3", null, 2, 1),
             |  ("1", null, 1, 1)
           """.stripMargin)

        val pairwiseStatMapAsJson = s"""{"$x": [["$y","1.0"]], "$y": [["$x","0.846950694324252"]]}"""
        val domainStatMapAsJson = s"""{"$tid": 9,"$x": 3,"$y": 4,"$z": 3}"""

        def testComputeDomain(minCorrThres: Double, domain_threshold_beta: Double, expected: Seq[Row]): Unit = {
          val domainDf = RepairApi.computeDomainInErrorCells(
            "inputView", "errCell", tid, s"$z", s"$x,$y", "freqAttrStats", pairwiseStatMapAsJson, domainStatMapAsJson, 4, minCorrThres, 0.0, domain_threshold_beta)
          assert(domainDf.columns.toSet === Set(tid, "attribute", "current_value", "domain"))
          val df = domainDf
            .selectExpr("*", "inline(domain)")
            .selectExpr(s"`$tid`", "attribute", "current_value", "n")
          checkAnswer(df, expected)
        }

        testComputeDomain(0.0, 0.01, Seq(
          Row(1, s"$x", "2", "1"),
          Row(1, s"$x", "2", "2"),
          Row(1, s"$x", "2", "3"),
          Row(3, s"$y", "test-3", "test-1"),
          Row(3, s"$y", "test-3", "test-3"),
          Row(6, s"$y", "test-2", "test-1"),
          Row(6, s"$y", "test-2", "test-2"),
          Row(6, s"$y", "test-2", "test-2a")
        ))
      }
    }
  }
}
