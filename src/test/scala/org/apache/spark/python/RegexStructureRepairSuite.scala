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

package org.apache.spark.python

import org.apache.spark.SparkFunSuite

class RegexStructureRepairSuite extends SparkFunSuite {

  test("basic parsing tests") {
    assert(RegexParser.parse("^[0-9]{1,3} patients$") === Seq(
      (RegexTokenType.Other, "^"),
      (RegexTokenType.Pattern, "[0-9]{1,3}"),
      (RegexTokenType.Constant, " patients"),
      (RegexTokenType.Other, "$")
    ))
    assert(RegexParser.parse("^[0-9]{1,3}%$") === Seq(
      (RegexTokenType.Other, "^"),
      (RegexTokenType.Pattern, "[0-9]{1,3}"),
      (RegexTokenType.Constant, "%"),
      (RegexTokenType.Other, "$")
    ))
  }

  test("repair-based structural repair") {
    Seq(
      ("^[0-9]{1,3} patients$", Seq(
        ("32 patixxts", Some("32 patients")),
        ("x2 patixxts", None))),
      ("^[0-9]{1,3}%", Seq(
        ("33x", Some("33%")),
        ("x2%", None))),
      ("^[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}$", Seq(
        ("23.39.23.11", Some("23-39-23-11")),
        ("23.x9.2x.1x", None)))).foreach { case (regex, tests) =>
      val repair = RegexStructureRepair(regex)
      tests.foreach { case (input, expected) =>
        assert(repair(input) === expected)
      }
    }
  }
}
