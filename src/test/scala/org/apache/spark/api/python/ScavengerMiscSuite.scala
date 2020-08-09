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

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

class ScavengerMiscSuite extends QueryTest with SharedSparkSession{

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sql(s"SET ${SQLConf.CBO_ENABLED.key}=true")
  }

  test("flattenTable") {
    withTable("t") {
      spark.sql("CREATE TABLE t(tid STRING, v1 INT, v2 STRING) USING parquet")
      spark.sql(
        s"""
           |INSERT INTO t VALUES
           |  ("1", 100000, "test-1"),
           |  ("2", 200000, "test-2"),
           |  ("3", 300000, "test-3"),
           |  ("4", 400000, "test-4")
         """.stripMargin)

      val df = ScavengerMiscApi.flattenTable("default", "t", "tid")
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
  }
}
