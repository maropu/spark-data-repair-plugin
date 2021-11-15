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

import java.io.File
import java.nio.charset.StandardCharsets

import com.google.common.io.Files

import org.apache.spark.sql.{AnalysisException, DataFrame, QueryTest, Row}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

class ErrorDetectorSuite extends QueryTest with SharedSparkSession {

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sql(s"SET ${SQLConf.CBO_ENABLED.key}=true")
  }

  private def resourcePath(f: String): String = {
    Thread.currentThread().getContextClassLoader.getResource(f).getPath
  }

  test("Error detector - common error handling") {
    Seq[String => DataFrame](ErrorDetectorApi.detectNullCells(_, "tid", ""),
      ErrorDetectorApi.detectErrorCellsFromRegEx(_, "tid", "", null),
      ErrorDetectorApi.detectErrorCellsFromConstraints(_, "tid", "", null),
      ErrorDetectorApi.detectErrorCellsFromOutliers(_, "tid", "")
    ).foreach { f =>
      val errMsg = intercept[AnalysisException] { f("nonexistent") }.getMessage()
      assert(errMsg.contains("Table or view not found: nonexistent"))
    }
  }

  test("Error detector - common error handling for non-existent target attributes") {
    withTable("t") {
      spark.sql("CREATE TABLE t(tid STRING, v1 STRING) USING parquet")
      Seq[String => DataFrame](ErrorDetectorApi.detectNullCells(_, "tid", "v2,v3"),
        ErrorDetectorApi.detectErrorCellsFromRegEx(_, "tid", "v2,v3", null),
        ErrorDetectorApi.detectErrorCellsFromConstraints(_, "tid", "v2,v3", null),
        ErrorDetectorApi.detectErrorCellsFromOutliers(_, "tid", "v2,v3")
      ).foreach { f =>
        val errMsg = intercept[AnalysisException] { f("t") }.getMessage()
        assert(errMsg.contains("Target attributes not found in t: v2,v3"))
      }
    }
  }

  test("NULL-based error detector") {
    withTable("t") {
      spark.sql("CREATE TABLE t(tid STRING, v1 INT, v2 DOUBLE, v3 STRING) USING parquet")
      spark.sql(
        s"""
           |INSERT INTO t VALUES
           |  ("1", 100000,  3.0, "test-1"),
           |  ("2",   NULL,  8.0, "test-2"),
           |  ("3", 300000,  1.0,     NULL),
           |  ("4", 400000, NULL, "test-4")
         """.stripMargin)

      def test(targetAttrList: String, expected: Seq[Row]): Unit = {
        val df = ErrorDetectorApi.detectNullCells("default.t", "tid", targetAttrList)
        checkAnswer(df, expected)
      }
      test("", Row("2", "v1") :: Row("3", "v3") :: Row("4", "v2") :: Nil)
      test("v1", Row("2", "v1") :: Nil)
      test("v2,v3", Row("3", "v3") :: Row("4", "v2") :: Nil)
      test("v3,v1", Row("2", "v1") :: Row("3", "v3") :: Nil)
      test("v3,v2,v5", Row("3", "v3") :: Row("4", "v2") :: Nil)
    }
  }

  test("RegEx-based error detector") {
    withTable("t") {
      spark.sql("CREATE TABLE t(tid STRING, v1 INT, v2 DOUBLE, v3 STRING) USING parquet")
      spark.sql(
        s"""
           |INSERT INTO t VALUES
           |  ("1",       123,  53.0, "123-abc"),
           |  ("2",    123456, 123.0, "456-efg"),
           |  ("3",    123000, 456.0,      NULL),
           |  ("4", 987654321,  NULL, "123-hij")
         """.stripMargin)

      def test(targetAttrList: String, regex: String, cellsAsString: Boolean, expected: Seq[Row]): Unit = {
        val df = ErrorDetectorApi
          .detectErrorCellsFromRegEx("default.t", "tid", targetAttrList, regex, cellsAsString)
        checkAnswer(df, expected)
      }
      test("", "123-hij", false, Row("4", "v3") :: Nil)
      test("", "123.*", false, Row("1", "v3") :: Row("4", "v3") :: Nil)
      test("", "123.*", true,
        Row("1", "v1") :: Row("1", "v3") :: Row("2", "v1") :: Row("2", "v2") :: Row("3", "v1") :: Row("4", "v3") :: Nil)
      test("v3", "123.*", true,
        Row("1", "v3") :: Row("4", "v3") :: Nil)
      test("v2,v3", "123.*", true,
        Row("1", "v3") :: Row("4", "v3") :: Row("2", "v2") :: Nil)
      test("v2,v1", "123.*", true,
        Row("1", "v1") :: Row("2", "v1") :: Row("2", "v2") :: Row("3", "v1") :: Nil)
      test("v1,v4,v5", "123.*", true,
        Row("1", "v1") :: Row("2", "v1") :: Row("3", "v1") :: Nil)
    }
  }

  test("RegEx-based error detector - invalid regex") {
    withTempView("t") {
      spark.range(1).selectExpr("id AS tid", "1 AS value").createOrReplaceTempView("t")
      def test(regex: String): Unit = {
        val df = ErrorDetectorApi.detectErrorCellsFromRegEx("t", "tid", "", regex)
        checkAnswer(df, Nil)
      }
      test(null)
      test("")
      test("    ")
    }
  }

  test("Constraint-based error detector") {
    withTable("t") {
      spark.sql("CREATE TABLE t(tid STRING, v1 INT, v2 STRING) USING parquet")
      spark.sql(
        s"""
           |INSERT INTO t VALUES
           |  ("1", 1, "test-1"),
           |  ("2", 1, "test-1"),
           |  ("3", 1,     NULL),
           |  ("4", 2, "test-2"),
           |  ("5", 2, "test-X"),
           |  ("6", 3, "test-3"),
           |  ("7", 4, "test-4"),
           |  ("8", 4, "test-4")
         """.stripMargin)

      withTempDir { tempDir =>
        // This test assumes the table `t` has a function dependency: `v1` -> `v2`
        val constraintFilePath = s"${tempDir.getCanonicalPath}/constraints.txt"
        Files.write("t1&t2&EQ(t1.v1,t2.v1)&IQ(t1.v2,t2.v2)",
          new File(constraintFilePath),
          StandardCharsets.UTF_8)

        def test(targetAttrList: String, expected: Seq[Row]): Unit = {
          val df = ErrorDetectorApi
            .detectErrorCellsFromConstraints("default.t", "tid", targetAttrList, constraintFilePath)
          checkAnswer(df, expected)
        }
        test("",
          Row("1", "v1") ::
            Row("1", "v2") ::
            Row("2", "v1") ::
            Row("2", "v2") ::
            Row("3", "v1") ::
            Row("3", "v2") ::
            Row("4", "v1") ::
            Row("4", "v2") ::
            Row("5", "v1") ::
            Row("5", "v2") ::
            Nil)
        test("v1",
          Row("1", "v1") ::
            Row("2", "v1") ::
            Row("3", "v1") ::
            Row("4", "v1") ::
            Row("5", "v1") ::
            Nil)
        test("v2,v1",
          Row("1", "v1") ::
            Row("1", "v2") ::
            Row("2", "v1") ::
            Row("2", "v2") ::
            Row("3", "v1") ::
            Row("3", "v2") ::
            Row("4", "v1") ::
            Row("4", "v2") ::
            Row("5", "v1") ::
            Row("5", "v2") ::
            Nil)
        test("v2,v3",
            Row("1", "v2") ::
            Row("2", "v2") ::
            Row("3", "v2") ::
            Row("4", "v2") ::
            Row("5", "v2") ::
            Nil)
      }
    }
  }

  test("Constraint-based error detector - adult") {
    withTable("adult") {
      val adultFilePath = resourcePath("adult.csv")
      spark.read.option("header", true).format("csv").load(adultFilePath).write.saveAsTable("adult")
      val constraintFilePath = resourcePath("adult_constraints.txt")
      val df = ErrorDetectorApi
        .detectErrorCellsFromConstraints("default.adult", "tid", "", constraintFilePath)
      checkAnswer(df,
        Row("4", "Relationship") ::
        Row("4", "Sex") ::
        Row("11", "Relationship") ::
        Row("11", "Sex") ::
        Nil)
    }
  }

  test("Constraint-based error detector - invalid constraintFilePath") {
    withTempView("t") {
      spark.range(1).selectExpr("id AS tid", "1 AS value").createOrReplaceTempView("t")
      def test(constraintFilePath: String): Unit = {
        val df = ErrorDetectorApi.detectErrorCellsFromConstraints("t", "tid", "", constraintFilePath)
        checkAnswer(df, Nil)
      }
      test(null)
      test("")
      test("     ")
    }
  }

  test("Outlier-based error detector") {
    withTempView("t") {
      val df1 = spark.range(1000).selectExpr("id AS tid", "double(100.0) AS value")
      val df2 = spark.range(1).selectExpr("1000L AS tid", "double(0.0) AS value")
      df1.union(df2).createOrReplaceTempView("t")
      def test(targetAttrList: String, approxEnabled: Boolean, expected: Seq[Row]): Unit = {
        val resultDf = ErrorDetectorApi.detectErrorCellsFromOutliers("t", "tid", targetAttrList, approxEnabled)
        checkAnswer(resultDf, Row(1000L, "value"))
      }
      Seq(false, true).foreach { approxEnabled =>
        test("", approxEnabled, Row(1000L, "value") :: Nil)
        test("value", approxEnabled, Row(1000L, "value") :: Nil)
        test("v,value", approxEnabled, Row(1000L, "value") :: Nil)
      }
    }
  }
}
