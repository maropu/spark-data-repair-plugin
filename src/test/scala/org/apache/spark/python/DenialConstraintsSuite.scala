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

  val leftIdent = DenialConstraints.leftRelationIdent
  val rightIdent = DenialConstraints.rightRelationIdent

  test("constraint parsing - adult") {
    val constraints = parseConstraints(
      s"""t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")
         |t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")
       """.stripMargin)
    assert(constraints.references.toSet === Set("Sex", "Relationship"))
    assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === Set(
      Set(s"""$leftIdent.Sex <=> "Female"""", s"""$leftIdent.Relationship <=> "Husband""""),
      Set(s"""$leftIdent.Sex <=> "Male"""", s"""$leftIdent.Relationship <=> "Wife"""")))
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
    assert(constraints.references.toSet === Set(
      "HospitalOwner", "MeasureName", "Condition", "PhoneNumber", "CountyName", "ProviderNumber",
      "HospitalName", "HospitalType", "EmergencyService", "City", "ZipCode", "Address1",
      "State", "Stateavg", "MeasureCode"))
    assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === Set(
      Set(s"$leftIdent.ProviderNumber <=> $rightIdent.ProviderNumber",
        s"NOT($leftIdent.HospitalName <=> $rightIdent.HospitalName)"),
      Set(s"$leftIdent.Condition <=> $rightIdent.Condition",
        s"$leftIdent.MeasureName <=> $rightIdent.MeasureName",
        s"NOT($leftIdent.HospitalType <=> $rightIdent.HospitalType)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.Address1 <=> $rightIdent.Address1)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.PhoneNumber <=> $rightIdent.PhoneNumber)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.ProviderNumber <=> $rightIdent.ProviderNumber)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.ZipCode <=> $rightIdent.ZipCode)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"$leftIdent.PhoneNumber <=> $rightIdent.PhoneNumber",
        s"$leftIdent.HospitalOwner <=> $rightIdent.HospitalOwner",
        s"NOT($leftIdent.State <=> $rightIdent.State)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.City <=> $rightIdent.City)"),
      Set(s"$leftIdent.MeasureName <=> $rightIdent.MeasureName",
        s"NOT($leftIdent.MeasureCode <=> $rightIdent.MeasureCode)"),
      Set(s"$leftIdent.MeasureCode <=> $rightIdent.MeasureCode",
        s"NOT($leftIdent.MeasureName <=> $rightIdent.MeasureName)"),
      Set(s"$leftIdent.HospitalName <=> $rightIdent.HospitalName",
        s"NOT($leftIdent.HospitalOwner <=> $rightIdent.HospitalOwner)"),
      Set(s"$leftIdent.MeasureCode <=> $rightIdent.MeasureCode",
        s"NOT($leftIdent.Condition <=> $rightIdent.Condition)"),
      Set(s"$leftIdent.MeasureCode <=> $rightIdent.MeasureCode",
        s"NOT($leftIdent.Stateavg <=> $rightIdent.Stateavg)"),
      Set(s"$leftIdent.ZipCode <=> $rightIdent.ZipCode",
        s"NOT($leftIdent.EmergencyService <=> $rightIdent.EmergencyService)"),
      Set(s"$leftIdent.City <=> $rightIdent.City",
        s"NOT($leftIdent.CountyName <=> $rightIdent.CountyName)")))
  }
}
