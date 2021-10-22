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

import org.apache.spark.TestUtils

import java.io.File
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.util.fileToString
import org.apache.spark.sql.test.SharedSparkSession

class DepGraphSuite extends QueryTest with SharedSparkSession {

  private def resourcePath(f: String): String = {
    Thread.currentThread().getContextClassLoader.getResource(f).getPath
  }

  private def checkOutputString(actual: String, expected: String): Unit = {
    def normalize(s: String) = s.replaceAll(" ", "").replaceAll("\n", "")
    assert(normalize(actual) == normalize(expected),
      s"`$actual` didn't match an expected string `$expected`")
  }

  test("generateDepGraph") {
    withTempDir { dirPath =>
      withTempView("inputView") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW inputView(tid, x, y, z) AS SELECT * FROM VALUES
             |  (1, "2", "test-1", 1.0),
             |  (2, "2", "test-2", 1.0),
             |  (3, "3", "test-1", 3.0),
             |  (4, "2", "test-2", 2.0),
             |  (5, "1", "test-1", 1.0),
             |  (6, "2", "test-1", 1.0),
             |  (7, "3", "test-3", 2.0),
             |  (8, "3", "test-3", 3.0),
             |  (9, "2", "test-2a", 2.0)
           """.stripMargin)

        val path = s"${dirPath.getAbsolutePath}/d"
        val targetAttrs = Seq("tid", "x", "y", "z")
        DepGraph.generateDepGraph(path, "inputView", "svg", targetAttrs, 8, 100, 1.0, 0.0, false)

        val graphString = fileToString(new File(s"${dirPath.getAbsolutePath}/d/depgraph.dot"))
        checkOutputString(graphString,
          s"""
             |digraph {
             |  graph [pad="0.5", nodesep="0.5", ranksep="2", fontname="Helvetica"];
             |  node [shape=plain]
             |  rankdir=LR;
             |
             |  "x_1" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>x_1</i></td></tr>
             |    <tr><td port="0">1</td></tr>
             |  <tr><td port="1">2</td></tr>
             |  <tr><td port="2">3</td></tr>
             |  </table>>];
             |
             |
             |  "x_2" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>x_2</i></td></tr>
             |    <tr><td port="0">3</td></tr>
             |  <tr><td port="1">2</td></tr>
             |  <tr><td port="2">1</td></tr>
             |  </table>>];
             |
             |
             |  "y_0" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>y_0</i></td></tr>
             |    <tr><td port="0">test-1</td></tr>
             |  <tr><td port="1">test-2</td></tr>
             |  <tr><td port="2">test-3</td></tr>
             |  <tr><td port="3">test-2a</td></tr>
             |  </table>>];
             |
             |
             |  "y_4" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>y_4</i></td></tr>
             |    <tr><td port="0">test-3</td></tr>
             |  <tr><td port="1">test-2</td></tr>
             |  <tr><td port="2">test-1</td></tr>
             |  <tr><td port="3">test-2a</td></tr>
             |  </table>>];
             |
             |
             |  "z_3" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>z_3</i></td></tr>
             |    <tr><td port="0">3.0</td></tr>
             |  <tr><td port="1">2.0</td></tr>
             |  <tr><td port="2">1.0</td></tr>
             |  </table>>];
             |
             |
             |  "z_5" [label=<
             |  <table border="1" cellborder="0" cellspacing="0">
             |    <tr><td bgcolor="lightgray" port="nodeName"><i>z_5</i></td></tr>
             |    <tr><td port="0">2.0</td></tr>
             |  <tr><td port="1">3.0</td></tr>
             |  <tr><td port="2">1.0</td></tr>
             |  </table>>];
             |
             |  "x" [ shape="box" ];
             |  "x" [ shape="box" ];
             |  "y" [ shape="box" ];
             |  "y" [ shape="box" ];
             |  "z" [ shape="box" ];
             |  "z" [ shape="box" ];
             |  "x_1":nodeName -> "x" [ arrowhead="none" penwidth="2.0" ];
             |  "x_2":0 -> "z_3":0 [ color="gray33" penwidth="0.7309297535714574"  ];
             |  "x_2":0 -> "z_3":1 [ color="gray66" penwidth="0.1"  ];
             |  "x_2":1 -> "z_3":1 [ color="gray60" penwidth="0.7309297535714574"  ];
             |  "x_2":1 -> "z_3":2 [ color="gray40" penwidth="1.1"  ];
             |  "x_2":2 -> "z_3":2 [ color="gray0" penwidth="0.1"  ];
             |  "x_2":nodeName -> "x" [ arrowhead="none" penwidth="2.0" ];
             |  "y_0":0 -> "x_1":0 [ color="gray75" penwidth="0.1"  ];
             |  "y_0":0 -> "x_1":1 [ color="gray50" penwidth="0.9547556456757273"  ];
             |  "y_0":0 -> "x_1":2 [ color="gray75" penwidth="0.1"  ];
             |  "y_0":1 -> "x_1":1 [ color="gray0" penwidth="0.9547556456757273"  ];
             |  "y_0":2 -> "x_1":2 [ color="gray0" penwidth="0.9547556456757273"  ];
             |  "y_0":3 -> "x_1":1 [ color="gray0" penwidth="0.1"  ];
             |  "y_0":nodeName -> "y" [ arrowhead="none" penwidth="2.0" ];
             |  "y_4":0 -> "z_5":0 [ color="gray50" penwidth="0.1"  ];
             |  "y_4":0 -> "z_5":1 [ color="gray50" penwidth="0.1"  ];
             |  "y_4":1 -> "z_5":0 [ color="gray50" penwidth="0.1"  ];
             |  "y_4":1 -> "z_5":2 [ color="gray50" penwidth="0.1"  ];
             |  "y_4":2 -> "z_5":1 [ color="gray75" penwidth="0.1"  ];
             |  "y_4":2 -> "z_5":2 [ color="gray25" penwidth="1.4547556456757276"  ];
             |  "y_4":3 -> "z_5":0 [ color="gray0" penwidth="0.1"  ];
             |  "y_4":nodeName -> "y" [ arrowhead="none" penwidth="2.0" ];
             |  "z_3":nodeName -> "z" [ arrowhead="none" penwidth="2.0" ];
             |  "z_5":nodeName -> "z" [ arrowhead="none" penwidth="2.0" ];
             |}
           """.stripMargin)
      }
    }
  }

  test("graph image generation") {
    assume(TestUtils.testCommandAvailable("dot"))

    withTempDir { dirPath =>
      withTempView("inputView") {
        spark.sql(
          s"""
             |CREATE TEMPORARY VIEW inputView(tid, x, y, z) AS SELECT * FROM VALUES
             |  (1, "2", "test-1", 1.0),
             |  (2, "2", "test-2", 1.0),
             |  (3, "3", "test-1", 3.0),
             |  (4, "2", "test-2", 2.0),
             |  (5, "1", "test-1", 1.0),
             |  (6, "2", "test-1", 1.0),
             |  (7, "3", "test-3", 2.0),
             |  (8, "3", "test-3", 3.0),
             |  (9, "2", "test-2a", 2.0)
           """.stripMargin)

        val path = s"${dirPath.getAbsolutePath}/d"
        val targetAttrs = Seq("tid", "x", "y", "z")

        DepGraph.validImageFormatSet.foreach { format =>
          val outputPath = s"${dirPath.getAbsolutePath}/$format"
          DepGraph.generateDepGraph(outputPath, "inputView", format, targetAttrs, 8, 100, 1.0, 0.0, false)
          val imgFile = new File(s"$outputPath/depgraph.$format")
          assert(imgFile.exists())
        }
      }
    }
  }

  test("computeFunctionalDeps") {
    withTempView("hospital") {
      val hospitalFilePath = resourcePath("hospital.csv")
      spark.read.option("header", true).format("csv").load(hospitalFilePath).createOrReplaceTempView("hospital")
      val constraintFilePath = resourcePath("hospital_constraints.txt")
      val jsonString = DepGraph.computeFunctionalDeps("hospital", constraintFilePath)
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

      val jsonString = DepGraph.computeFunctionalDepMap("tempView", "x", "y")
      val jsonObj = parse(jsonString)
      val data = jsonObj.asInstanceOf[JObject].values
      assert(data === Map("3" -> "test-3", "1" -> "test-1"))
    }
  }
}
