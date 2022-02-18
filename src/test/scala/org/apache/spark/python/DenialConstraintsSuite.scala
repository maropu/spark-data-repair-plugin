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

class DenialConstraintsSuite extends SparkFunSuite {

  val leftIdent = DenialConstraints.leftRelationIdent
  val rightIdent = DenialConstraints.rightRelationIdent

  test("constraint parsing") {
    def testValidConstraintSyntax(c: String, preds: Set[String], refs: Set[String]): Unit = {
      val parsedPreds = DenialConstraints.parse(c)
      val parsedRefs = parsedPreds.flatMap(_.references).distinct
      assert(parsedPreds.map(_.toString).toSet === preds)
      assert(parsedRefs.toSet === refs)
    }
    testValidConstraintSyntax("""t1&EQ(t1.v1,"abc")&EQ(t1.v2,"def")""",
      Set(s"""$leftIdent.v1 <=> "abc"""", s"""$leftIdent.v2 <=> "def""""),
      Set("v1", "v2"))
    testValidConstraintSyntax("""t1&t2&EQ(t1.v1,t2.v1)&IQ(t1.v2,t2.v2)""",
      Set(s"""$leftIdent.v1 <=> $rightIdent.v1""", s"""NOT($leftIdent.v2 <=> $rightIdent.v2)"""),
      Set("v1", "v2"))
    testValidConstraintSyntax("""t1&t2&LT(t1.v1,t2.v1)&GT(t1.v2,t2.v2)&EQ(t1.v1,t2.v1)""",
      Set(s"""$leftIdent.v1 < $rightIdent.v1""", s"""$leftIdent.v2 > $rightIdent.v2""",
        s"""$leftIdent.v1 <=> $rightIdent.v1"""),
      Set("v1", "v2"))
    testValidConstraintSyntax(""" t1 & EQ ( t1.v1 , "abc") & EQ ( t1.v2 , "def" ) """,
      Set(s"""$leftIdent.v1 <=> "abc"""", s"""$leftIdent.v2 <=> "def""""),
      Set("v1", "v2"))
    testValidConstraintSyntax("""t1 & t2 & EQ ( t1.v1 , t2.v1 ) & IQ ( t1.v2 , t2.v2 ) """,
      Set(s"""$leftIdent.v1 <=> $rightIdent.v1""", s"""NOT($leftIdent.v2 <=> $rightIdent.v2)"""),
      Set("v1", "v2"))
  }

  test("constraint parsing - invalid cases") {
    def testInvalidConstraintSyntax(dc: String, expectedErrMsg: String): Unit = {
      val errMsg = intercept[IllegalArgumentException] {
        DenialConstraints.parse(dc)
      }.getMessage
      assert(errMsg.contains("Failed to parse an input string") ||
        errMsg.contains("Illegal predicates found") ||
        errMsg.contains("At least two predicate candidates should be given"))

      val logAppender = new LogAppender("invalid constraint parsing tests")
      withLogAppender(logAppender) {
        val c = DenialConstraints.parseAndVerifyConstraints(Seq(dc), "", Seq("v1", "v2"))
        assert(c.references === Nil)
        assert(c.predicates === Nil)
      }
      val messages = logAppender.loggingEvents
        .filter(_.getRenderedMessage.contains(expectedErrMsg))
      assert(messages.size === 1)
    }
    testInvalidConstraintSyntax("""EQ(t1.v1,"abc")""",
      """Illegal constraint format found: EQ(t1.v1,"abc")""")
    testInvalidConstraintSyntax("""1a&IQ(1a.v,"abc")""",
      """Illegal constraint format found: 1a&IQ(1a.v,"abc")""")
    testInvalidConstraintSyntax("""key&EQ(noexistent.v1,"abc")&EQ(key,"def")""",
      """Illegal constraint format found: key&EQ(noexistent.v1,"abc")&EQ(key,"def")""")
    testInvalidConstraintSyntax("""t1&1a&EQ(t1.v,"abc")&IQ(1a.v,"def")""",
      """Illegal constraint format found: t1&1a&EQ(t1.v,"abc")&IQ(1a.v,"def")""")
    testInvalidConstraintSyntax("""t1&EQ(t1.v1,"abc")&IL(t1.v1, "def")&EQ(t1.v2,"ghi")""",
      """Illegal constraint format found: t1&EQ(t1.v1,"abc")&IL(t1.v1, "def")&EQ(t1.v2,"ghi")""")
    testInvalidConstraintSyntax("""t1&t2&GT(t3.v0,"abc")&EQ(t1.v1,t2.v1)&IQ(t1.v2,t2.v2)""",
      """Illegal constraint format found: t1&t2&GT(t3.v0,"abc")&EQ(t1.v1,t2.v1)&IQ(t1.v2,t2.v2)""")
    testInvalidConstraintSyntax("""t1&t2&GT(t3.v0,"abc")&EQ(t1.v1,t2.v1)&IL(t1.v2,t2.v2)""",
      """Illegal constraint format found: t1&t2&GT(t3.v0,"abc")&EQ(t1.v1,t2.v1)&IL(t1.v2,t2.v2)""")
    testInvalidConstraintSyntax("""t1&EQ(t1.v1,"abc")""",
      """Illegal constraint format found: t1&EQ(t1.v1,"abc")""")
    testInvalidConstraintSyntax("""t1&""",
      """Illegal constraint format found: t1&""")
    testInvalidConstraintSyntax("""t1""",
      """Illegal constraint format found: t1""")
    testInvalidConstraintSyntax("""a&b&""",
      """Illegal constraint format found: a&b&""")
    testInvalidConstraintSyntax("""k1&k2""",
      """Illegal constraint format found: k1&k2""")
  }

  test("constraint parsing - adult") {
    val constraints = DenialConstraints.parseAndVerifyConstraints(Seq(
      """t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")""",
      """t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")"""),
      "adult",
      Seq("Sex", "Relationship"))
    assert(constraints.references.toSet === Set("Sex", "Relationship"))
    assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === Set(
      Set(s"""$leftIdent.Sex <=> "Female"""", s"""$leftIdent.Relationship <=> "Husband""""),
      Set(s"""$leftIdent.Sex <=> "Male"""", s"""$leftIdent.Relationship <=> "Wife"""")))
  }

  test("constraint parsing - hospital") {
    val constraints = DenialConstraints.parseAndVerifyConstraints(Seq(
        s"""t1&t2&EQ(t1.Condition,t2.Condition)&EQ(t1.MeasureName,t2.MeasureName)&IQ(t1.HospitalType,t2.HospitalType)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.ZipCode,t2.ZipCode)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.PhoneNumber,t2.PhoneNumber)""",
        s"""t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.MeasureName,t2.MeasureName)""",
        s"""t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.Stateavg,t2.Stateavg)""",
        s"""t1&t2&EQ(t1.ProviderNumber,t2.ProviderNumber)&IQ(t1.HospitalName,t2.HospitalName)""",
        s"""t1&t2&EQ(t1.MeasureCode,t2.MeasureCode)&IQ(t1.Condition,t2.Condition)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.Address1,t2.Address1)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.HospitalOwner,t2.HospitalOwner)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.ProviderNumber,t2.ProviderNumber)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&EQ(t1.PhoneNumber,t2.PhoneNumber)&EQ(t1.HospitalOwner,t2.HospitalOwner)&IQ(t1.State,t2.State)""",
        s"""t1&t2&EQ(t1.City,t2.City)&IQ(t1.CountyName,t2.CountyName)""",
        s"""t1&t2&EQ(t1.ZipCode,t2.ZipCode)&IQ(t1.EmergencyService,t2.EmergencyService)""",
        s"""t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.City,t2.City)""",
        s"""t1&t2&EQ(t1.MeasureName,t2.MeasureName)&IQ(t1.MeasureCode,t2.MeasureCode)"""),
      "hospital",
      Seq("HospitalOwner", "MeasureName", "Condition", "PhoneNumber", "CountyName", "ProviderNumber",
        "HospitalName", "HospitalType", "EmergencyService", "City", "ZipCode", "Address1",
        "State", "Stateavg", "MeasureCode"))
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

  test("simple constraint parsing") {
    def testValidConstraintSyntax(cs: String, preds: Set[Set[String]], refs: Set[String]): Unit = {
      val parsedPreds = cs.split(";").map(_.trim()).filter(_.nonEmpty).map(DenialConstraints.parseAlt).toSeq
      val parsedRefs = parsedPreds.flatMap(_.flatMap(_.references)).distinct
      assert(parsedPreds.map(_.map(_.toString).toSet).toSet === preds)
      assert(parsedRefs.toSet === refs)
    }
    testValidConstraintSyntax("X->Y;Y->Z",
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)"),
        Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("X", "Y", "Z"))
    testValidConstraintSyntax("",
      Set.empty,
      Set.empty)
    testValidConstraintSyntax("X->Y;Y->Z",
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)"),
        Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("X", "Y", "Z"))
    testValidConstraintSyntax("X->Y;Y->Z;",
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)"),
        Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("X", "Y", "Z"))
    testValidConstraintSyntax(";X ->  Y; Y-> Z   ",
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)"),
        Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("X", "Y", "Z"))
  }

  test("simple constraint parsing - verification") {
    def testValidConstraintSyntax(
        input: String, inputAttrs: Seq[String], preds: Set[Set[String]], refs: Set[String]): Unit = {
      val cs = input.split(";").map(_.trim()).filter(_.nonEmpty).toSeq
      val constraints = DenialConstraints.parseAndVerifyConstraints(cs, "", inputAttrs)
      assert(constraints.predicates.map(_.map(_.toString).toSet).toSet === preds)
      assert(constraints.references.toSet === refs)
    }
    testValidConstraintSyntax("X->Y;Y->Z", Seq("X", "Y", "Z"),
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)"),
        Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("X", "Y", "Z"))
    testValidConstraintSyntax("X->Y;Y->Z", Seq("X", "Y"),
      Set(Set(s"$leftIdent.X <=> $rightIdent.X", s"NOT($leftIdent.Y <=> $rightIdent.Y)")),
      Set("X", "Y"))
    testValidConstraintSyntax("X->Y;Y->Z", Seq("Y", "Z"),
      Set(Set(s"$leftIdent.Y <=> $rightIdent.Y", s"NOT($leftIdent.Z <=> $rightIdent.Z)")),
      Set("Y", "Z"))
    testValidConstraintSyntax("X->Y;Y->Z", Seq("Y"),
      Set.empty,
      Set.empty)
  }

  test("simple constraint parsing - invalid cases") {
    def testInvalidConstraintSyntax(s: String, expectedErrMsgs: Seq[String]): Unit = {
      val constraints = s.split(";").map(_.trim()).filter(_.nonEmpty)
      constraints.foreach { s =>
        val errMsg = intercept[IllegalArgumentException] {
          DenialConstraints.parseAlt(s)
        }.getMessage
        assert(errMsg.contains("Failed to parse an input string"))
      }
      val logAppender = new LogAppender("invalid constraint parsing tests")
      withLogAppender(logAppender) {
        val c = DenialConstraints.parseAndVerifyConstraints(constraints, "", Seq("v1", "v2"))
        assert(c.references === Nil)
        assert(c.predicates === Nil)
      }
      expectedErrMsgs.foreach { expectedErrMsg =>
        val messages = logAppender.loggingEvents
          .filter(_.getRenderedMessage.contains(expectedErrMsg))
        assert(messages.size === 1)
      }
    }
    testInvalidConstraintSyntax("""X=>Y;A""", Seq(
      """Illegal constraint format found: X=>Y""",
      """Illegal constraint format found: A""",
    ))
    testInvalidConstraintSyntax("""X- >Y;A=>B;;""", Seq(
      """Illegal constraint format found: X- >Y""",
      """Illegal constraint format found: A=>B""",
    ))
    testInvalidConstraintSyntax("""A ;X -<  Y; B =>  C; Y- > Z  """, Seq(
      """Illegal constraint format found: A""",
      """Illegal constraint format found: X -<  Y""",
      """Illegal constraint format found: B =>  C""",
      """Illegal constraint format found: Y- > Z"""
    ))
  }
}
