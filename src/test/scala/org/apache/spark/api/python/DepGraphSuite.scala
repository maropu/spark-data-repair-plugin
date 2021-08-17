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

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class DepGraphSuite extends QueryTest with SharedSparkSession {

  private def resourcePath(f: String): String = {
    Thread.currentThread().getContextClassLoader.getResource(f).getPath
  }

  test("computeFunctionalDeps") {
    withTempView("hospital") {
      val hospitalFilePath = resourcePath("hospital.csv")
      spark.read.option("header", true).format("csv").load(hospitalFilePath).createOrReplaceTempView("hospital")
      val constraintFilePath = resourcePath("hospital_constraints.txt")
      val jsonString = DepGraphApi.computeFunctionalDeps("hospital", constraintFilePath)
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

      val jsonString = DepGraphApi.computeFunctionDepMap("tempView", "x", "y")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data === Map("3" -> "test-3", "1" -> "test-1"))
    }
  }
}
