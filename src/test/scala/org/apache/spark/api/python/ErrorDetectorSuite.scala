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

import org.apache.spark.SparkException
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
    Seq[String => DataFrame](ErrorDetectorApi.detectNullCells("", _, "tid"),
      ErrorDetectorApi.detectErrorCellsFromRegEx("", _, "tid", null),
      ErrorDetectorApi.detectErrorCellsFromConstraints("", _, "tid", null),
      ErrorDetectorApi.detectErrorCellsFromOutliers("", _, "tid")
    ).foreach { f =>
      val errMsg = intercept[AnalysisException] { f("nonexistent") }.getMessage()
      assert(errMsg.contains("Table or view not found: nonexistent"))
    }

    withTempView("t") {
      spark.range(1).createOrReplaceTempView("t")

      Seq[String => DataFrame](ErrorDetectorApi.detectNullCells("", "t", _),
        ErrorDetectorApi.detectErrorCellsFromRegEx("", "t", _, null),
        ErrorDetectorApi.detectErrorCellsFromConstraints("", "t", _, null),
        ErrorDetectorApi.detectErrorCellsFromOutliers("", "t", _)
      ).foreach { f =>
        val errMsg1 = intercept[SparkException] { f("nonexistent") }.getMessage()
        assert(errMsg1.contains("Column 'nonexistent' does not exist in 't'"))

        val errMsg2 = intercept[SparkException] { f("id") }.getMessage()
        assert(errMsg2.contains("At least one valid column needs to exist, " +
          "but only one column 'id' exists"))
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

      val df = ErrorDetectorApi.detectNullCells("default", "t", "tid")
      checkAnswer(df,
        Row("2", "v1") :: Row("3", "v3") :: Row("4", "v2") :: Nil)
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

      val df1 = ErrorDetectorApi
        .detectErrorCellsFromRegEx("default", "t", "tid", "123-hij", cellsAsString = false)
      checkAnswer(df1, Row("4", "v3"))
      val df2 = ErrorDetectorApi
        .detectErrorCellsFromRegEx("default", "t", "tid", "123.*", cellsAsString = false)
      checkAnswer(df2,
        Row("1", "v3") ::
        Row("4", "v3") ::
        Nil)
      val df3 = ErrorDetectorApi
        .detectErrorCellsFromRegEx("default", "t", "tid", "123.*", cellsAsString = true)
      checkAnswer(df3,
        Row("1", "v1") ::
        Row("1", "v3") ::
        Row("2", "v1") ::
        Row("2", "v2") ::
        Row("3", "v1") ::
        Row("4", "v3") ::
        Nil)
    }
  }

  test("RegEx-based error detector - common error handling") {
    withTempView("t") {
      spark.range(1).selectExpr("id AS tid", "1 AS value").createOrReplaceTempView("t")
      val df1 = ErrorDetectorApi.detectErrorCellsFromRegEx("", "t", "tid", null)
      checkAnswer(df1, Nil)
      val df2 = ErrorDetectorApi.detectErrorCellsFromRegEx("", "t", "tid", "")
      checkAnswer(df2, Nil)
      val df3 = ErrorDetectorApi.detectErrorCellsFromRegEx("", "t", "tid", "    ")
      checkAnswer(df3, Nil)
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

        val df = ErrorDetectorApi
          .detectErrorCellsFromConstraints("default", "t", "tid", constraintFilePath)
        checkAnswer(df,
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
      }
    }
  }

  test("Constraint-based error detector - adult") {
    withTable("adult") {
      val adultFilePath = resourcePath("adult.csv")
      spark.read.option("header", true).format("csv").load(adultFilePath).write.saveAsTable("adult")
      val constraintFilePath = resourcePath("adult_constraints.txt")
      val df = ErrorDetectorApi
        .detectErrorCellsFromConstraints("default", "adult", "tid", constraintFilePath)
      checkAnswer(df,
        Row("4", "Relationship") ::
        Row("4", "Sex") ::
        Row("11", "Relationship") ::
        Row("11", "Sex") ::
        Nil)
    }
  }

  test("Constraint-based error detector - common error handling") {
    withTempView("t") {
      spark.range(1).selectExpr("id AS tid", "1 AS value").createOrReplaceTempView("t")
      val df1 = ErrorDetectorApi.detectErrorCellsFromConstraints("", "t", "tid", null)
      checkAnswer(df1, Nil)
      val df2 = ErrorDetectorApi.detectErrorCellsFromConstraints("", "t", "tid", "")
      checkAnswer(df2, Nil)
      val df3 = ErrorDetectorApi.detectErrorCellsFromConstraints("", "t", "tid", "    ")
      checkAnswer(df3, Nil)
    }
  }

  test("Outlier-based error detector") {
    withTempView("t") {
      val df1 = spark.range(1000).selectExpr("id AS tid", "double(100.0) AS value")
      val df2 = spark.range(1).selectExpr("1000L AS tid", "double(0.0) AS value")
      df1.union(df2).createOrReplaceTempView("t")
      Seq(true, false).foreach { approxEnabled =>
        val resultDf = ErrorDetectorApi
          .detectErrorCellsFromOutliers("", "t", "tid", approxEnabled)
        checkAnswer(resultDf, Row(1000L, "value"))
      }
    }
  }
}
