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

package org.apache.spark.util

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.util.RepairUtils._

class RepairUtilsSuite extends QueryTest with SharedSparkSession {

  test("checkSchema") {
    withTempView("tempView") {
      spark.range(1).selectExpr(
        "id tid", "int(1) a", "bigint(1) b", "string('a') c", "double(1.1) d"
        ).createOrReplaceTempView("tempView")
      assert(checkSchema(
        "tempView", "a int, b bigint, c string, d double", "tid", strict = true))
      assert(checkSchema(
        "tempView", "b bigint, a int, d double, c string", "tid", strict = true))
      assert(!checkSchema(
        "tempView", "a int, b bigint, c string, d double", "xid", strict = true))
      assert(!checkSchema(
        "tempView", "b bigint, a int, d double, c string, e tinyint", "tid", strict = true))
      assert(checkSchema(
        "tempView", "b bigint, a int", "tid", strict = false))
      assert(checkSchema(
        "tempView", "b bigint, a int", "tid", strict = true))
    }
  }
}
