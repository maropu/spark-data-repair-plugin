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

import org.apache.spark.sql.{AnalysisException, QueryTest, Row}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.util.RepairUtils._

class RepairMiscSuite extends QueryTest with SharedSparkSession {

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sql(s"SET ${SQLConf.CBO_ENABLED.key}=true")
    spark.sql("CREATE TABLE t(tid STRING, v1 INT, v2 STRING) USING parquet")
    spark.sql(
      s"""
         |INSERT INTO t VALUES
         |  ("1", 100000, "test-1"),
         |  ("2", 200000, "test-2"),
         |  ("3", 300000, "test-3"),
         |  ("4", 400000, "test-4")
       """.stripMargin)
  }

  protected override def afterAll(): Unit = {
    try {
      spark.sql("DROP TABLE IF EXISTS t")
    } finally {
      super.afterAll()
    }
  }

  test("flattenTable") {
    val df = RepairMiscApi.flattenTable("default", "t", "tid")
    val expectedSchema = "`tid` STRING,`attribute` STRING NOT NULL,`value` STRING"
    assert(df.schema.toDDL === expectedSchema)
    checkAnswer(df,
      Row("1", "v1", "100000") ::
      Row("2", "v1", "200000") ::
      Row("3", "v1", "300000") ::
      Row("4", "v1", "400000") ::
      Row("1", "v2", "test-1") ::
      Row("2", "v2", "test-2") ::
      Row("3", "v2", "test-3") ::
      Row("4", "v2", "test-4") ::
      Nil
    )
  }

  test("q-gram") {
    val res1 = RepairMiscApi.computeQgram(2, Seq("abcdef", "ghijklm", "n", "op", "qrs", "tuvwxyz"))
    assert(res1 === Seq("ab", "bc", "cd", "de", "ef", "gh", "hi", "ij", "jk", "kl", "lm", "n",
      "op", "qr", "rs", "tu", "uv", "vw", "wx", "xy", "yz"))
    val res2 = RepairMiscApi.computeQgram(3, Seq("abcdef", "ghijklm", "n", "op", "qrs", "tuvwxyz"))
    assert(res2 === Seq("abc", "bcd", "cde", "def", "ghi", "hij", "ijk", "jkl", "klm", "n",
      "op", "qrs", "tuv", "uvw", "vwx", "wxy", "xyz"))
    val res3 = RepairMiscApi.computeQgram(4, Seq("abcdef", "ghijklm", "n", "op", "qrs", "tuvwxyz"))
    assert(res3 === Seq("abcd", "bcde", "cdef", "ghij", "hijk", "ijkl", "jklm", "n",
      "op", "qrs", "tuvw", "uvwx", "vwxy", "wxyz"))

    val errMsg = intercept[IllegalArgumentException] {
      RepairMiscApi.computeQgram(0, Nil)
    }.getMessage
    assert(errMsg.contains("`q` must be positive, but 0 got"))
  }

  test("splitInputTableInto") {
    val df1 = RepairMiscApi.splitInputTableInto(2, "default", "t", "tid", "", "")
    val expectedSchema = "`tid` STRING,`k` INT NOT NULL"
    assert(df1.schema.toDDL === expectedSchema)
    assert(df1.select("k").collect.map(_.getInt(0)).toSet === Set(0, 1))
    val df2 = RepairMiscApi.splitInputTableInto(2, "default", "t", "tid", "v1", "")
    assert(df2.schema.toDDL === expectedSchema)
    assert(df2.select("k").collect.map(_.getInt(0)).toSet === Set(0, 1))

    val errMsg = intercept[AnalysisException] {
      RepairMiscApi.splitInputTableInto(2, "default", "t", "tid", "non-existent", "")
    }.getMessage
    assert(errMsg.contains("Columns 'non-existent' do not exist in 'default.t'"))
  }

  test("injectNullAt") {
    val df1 = RepairMiscApi.injectNullAt("default", "t", "v1", 1.0)
    val expectedSchema = "`tid` STRING,`v1` INT,`v2` STRING"
    assert(df1.schema.toDDL === expectedSchema)
    checkAnswer(df1,
      Row("1", null, "test-1") ::
      Row("2", null, "test-2") ::
      Row("3", null, "test-3") ::
      Row("4", null, "test-4") ::
      Nil
    )
    val df2 = RepairMiscApi.injectNullAt("default", "t", "v2", 0.0)
    assert(df2.schema.toDDL === expectedSchema)
    checkAnswer(df2,
      Row("1", 100000, "test-1") ::
      Row("2", 200000, "test-2") ::
      Row("3", 300000, "test-3") ::
      Row("4", 400000, "test-4") ::
      Nil
    )

    val errMsg = intercept[AnalysisException] {
      RepairMiscApi.injectNullAt("default", "t", "non-existent", 1.0)
    }.getMessage
    assert(errMsg.contains("Columns 'non-existent' do not exist in 'default.t'"))
  }

  test("repairAttrsFrom") {
    withTempView("inputView", "repairUpdates") {
      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW inputView(tid, x, y, z) AS SELECT * FROM VALUES
           |  (1, null, "test-1", 1.0D),
           |  (2, null, null, 2.0D),
           |  (3, 1, "test-2", null)
           """.stripMargin)

      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW repairUpdates(tid, attribute, repaired) AS SELECT * FROM VALUES
           |  (1, "x", "2.4"),
           |  (2, "x", "2.6"),
           |  (2, "y", "test-3"),
           |  (3, "z", "3.1D")
           """.stripMargin)

      val df = RepairMiscApi.repairAttrsFrom("repairUpdates", "", "inputView", "tid")
      checkAnswer(df, Seq(
        Row(1, 2, "test-1", 1.0D),
        Row(2, 3, "test-3", 2.0D),
        Row(3, 1, "test-2", 3.1D)))

      val errMsg = intercept[AnalysisException] {
        RepairMiscApi.repairAttrsFrom("inputView", "", "inputView", "tid")
      }.getMessage()
      assert(errMsg.contains("Table 'inputView' must have 'tid', 'attribute', and 'repaired' columns"))
    }
  }

  test("convertToHistogram") {
    withTempView("tempView") {
      spark.sql(
        s"""
           |CREATE TEMPORARY VIEW tempView(tid, x, y, z) AS SELECT * FROM VALUES
           |  (1, "1", "test-1", 1.0),
           |  (2, null, "test-1", 2.0),
           |  (3, "1", "test-2", 1.0)
         """.stripMargin)

      val df = RepairMiscApi.convertToHistogram("x,y,z", "", "tempView")
      assert(df.columns.toSet === Set("attribute", "histogram"))
      checkAnswer(df.selectExpr("attribute", "inline(histogram)"), Seq(
        Row("x", "1", 2L),
        Row("y", "test-1", 2L),
        Row("y", "test-2", 1L),
        Row("z", "1.0", 2L),
        Row("z", "2.0", 1L)))
    }
  }

  test("toErrorMap") {
    withTempView("errCellView", "IllegalView") {
      import testImplicits._
      Seq(("1", "v1"), ("2", "v1"), ("4", "v2")).toDF("tid", "attribute")
        .createOrReplaceTempView("errCellView")
      val df = RepairMiscApi.toErrorMap("errCellView", "default", "t", "tid")
      val expectedSchema = "`tid` STRING,`error_map` STRING NOT NULL"
      assert(df.schema.toDDL === expectedSchema)
      checkAnswer(df.selectExpr("error_map"),
        Row("*-") ::
        Row("*-") ::
        Row("-*") ::
        Row("--") ::
        Nil
      )

      val errMsg = intercept[AnalysisException] {
        createEmptyTable("tid STRING, illegal STRING").createOrReplaceTempView("IllegalView")
        RepairMiscApi.toErrorMap("IllegalView", "default", "t", "tid")
      }.getMessage
      assert(errMsg.contains("'IllegalView' must have 'tid' and 'attribute' columns"))
    }
  }
}
