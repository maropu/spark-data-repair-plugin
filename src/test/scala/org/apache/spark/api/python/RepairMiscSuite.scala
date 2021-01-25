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
import org.apache.spark.sql.{QueryTest, Row}
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
    val expectedSchema = "`tid` STRING,`attribute` STRING,`value` STRING"
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
    val expectedSchema = "`tid` STRING,`k` INT"
    assert(df1.schema.toDDL === expectedSchema)
    assert(df1.select("k").collect.map(_.getInt(0)).toSet === Set(0, 1))
    val df2 = RepairMiscApi.splitInputTableInto(2, "default", "t", "tid", "v1", "")
    assert(df2.schema.toDDL === expectedSchema)
    assert(df2.select("k").collect.map(_.getInt(0)).toSet === Set(0, 1))

    val errMsg = intercept[SparkException] {
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

    val errMsg = intercept[SparkException] {
      RepairMiscApi.injectNullAt("default", "t", "non-existent", 1.0)
    }.getMessage
    assert(errMsg.contains("Columns 'non-existent' do not exist in 'default.t'"))
  }

  test("toErrorMap") {
    withTempView("errCellView", "IllegalView") {
      import testImplicits._
      Seq(("1", "v1"), ("2", "v1"), ("4", "v2")).toDF("tid", "attribute")
        .createOrReplaceTempView("errCellView")
      val df = RepairMiscApi.toErrorMap("errCellView", "default", "t", "tid")
      val expectedSchema = "`tid` STRING,`error_map` STRING"
      assert(df.schema.toDDL === expectedSchema)
      checkAnswer(df.selectExpr("error_map"),
        Row("*-") ::
        Row("*-") ::
        Row("-*") ::
        Row("--") ::
        Nil
      )

      val errMsg = intercept[SparkException] {
        createEmptyTable("tid STRING, illegal STRING").createOrReplaceTempView("IllegalView")
        RepairMiscApi.toErrorMap("IllegalView", "default", "t", "tid")
      }.getMessage
      assert(errMsg.contains("'IllegalView' must have 'tid', 'attribute' columns"))
    }
  }
}
