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

import java.io.File
import java.nio.charset.StandardCharsets

import com.google.common.io.Files

import org.apache.spark.SparkFunSuite
import org.apache.spark.util.Utils

class DenialConstraintsSuite extends SparkFunSuite {

  private def parseConstraints(constraints: String): DenialConstraints = {
    val tempDir = Utils.createTempDir()
    val constraintFilePath = s"${tempDir.getCanonicalPath}/constraints.txt"
    Files.write(constraints, new File(constraintFilePath), StandardCharsets.UTF_8)
    DenialConstraints.parse(constraintFilePath)
  }

  test("constraint parsing - adult") {
    val constraints = parseConstraints(
      s"""t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")
         |t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")
       """.stripMargin)
    assert(constraints.leftTable === "t1")
    assert(constraints.rightTable === "__auto_generated_2")
    assert(constraints.references.toSet === Set("Sex", "Relationship"))
    assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === Set(
      Set("""t1.Sex <=> "Female"""", """t1.Relationship <=> "Husband""""),
      Set("""t1.Sex <=> "Male"""", """t1.Relationship <=> "Wife"""")))
  }

  test("constraint parsing - hospital") {
    val constraints = parseConstraints(
      s"""t1&t2&EQ(t1.Condition,t2.Condition)&EQ(t1.MeasureName,t2.MeasureName)&IQ(t1.HospitalType,t2.HospitalType)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.ZipCode,t2.ZipCode)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.PhoneNumber,t2.PhoneNumber)
         |t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.MeasureName,t2.MeasureName)
         |t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.Stateavg,t2.Stateavg)
         |t1&t2&EQ(t1.ProviderNumber,t2.ProviderNumber)&IQ(t1.HospitalName,t2.HospitalName)
         |t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.Condition,t2.Condition)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.Address1,t2.Address1)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.HospitalOwner,t2.HospitalOwner)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.ProviderNumber,t2.ProviderNumber)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&EQ(t1.PhoneNumber,t2.PhoneNumber)&EQ(t1.HospitalOwner,t2.HospitalOwner)&IQ(t1.State,t2.State)
         |t1&t2&EQ(t1.City,t2.City)&IQ(t1.CountyName,t2.CountyName)
         |t1&t2&EQ(t1.ZipCode,t2.ZipCode)&IQ(t1.EmergencyService,t2.EmergencyService)
         |t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.City,t2.City)
         |t1&t2&EQ(t1.MeasureName,t2.MeasureName)&IQ(t1.MeasureCode,t2.MeasureCode)
       """.stripMargin)
    assert(constraints.leftTable === "t1")
    assert(constraints.rightTable === "t2")
    assert(constraints.references.toSet === Set(
      "HospitalOwner", "MeasureName", "Condition", "PhoneNumber", "CountyName", "ProviderNumber",
      "HospitalName", "HospitalType", "EmergencyService", "City", "ZipCode", "Address1",
      "State", "Stateavg", "MeasureCode"))
    assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === Set(
      Set("t1.ProviderNumber <=> t2.ProviderNumber", "NOT(t1.HospitalName <=> t2.HospitalName)"),
      Set("t1.Condition <=> t2.Condition", "t1.MeasureName <=> t2.MeasureName",
        "NOT(t1.HospitalType <=> t2.HospitalType)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.Address1 <=> t2.Address1)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.PhoneNumber <=> t2.PhoneNumber)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.ProviderNumber <=> t2.ProviderNumber)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.ZipCode <=> t2.ZipCode)"),
      Set("t1.HospitalName <=> t2.HospitalName", "t1.PhoneNumber <=> t2.PhoneNumber",
        "t1.HospitalOwner <=> t2.HospitalOwner", "NOT(t1.State <=> t2.State)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.City <=> t2.City)"),
      Set("t1.MeasureName <=> t2.MeasureName", "NOT(t1.MeasureCode <=> t2.MeasureCode)"),
      Set("t1.MeasureCode <=> t2.MeasureCode", "NOT(t1.MeasureName <=> t2.MeasureName)"),
      Set("t1.HospitalName <=> t2.HospitalName", "NOT(t1.HospitalOwner <=> t2.HospitalOwner)"),
      Set("t1.MeasureCode <=> t2.MeasureCode", "NOT(t1.Condition <=> t2.Condition)"),
      Set("t1.MeasureCode <=> t2.MeasureCode", "NOT(t1.Stateavg <=> t2.Stateavg)"),
      Set("t1.ZipCode <=> t2.ZipCode", "NOT(t1.EmergencyService <=> t2.EmergencyService)"),
      Set("t1.City <=> t2.City", "NOT(t1.CountyName <=> t2.CountyName)")))
  }
}
