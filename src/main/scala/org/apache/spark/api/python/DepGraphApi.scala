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

import java.net.URI

import scala.collection.mutable
import scala.io.Source

import org.apache.spark.python.DenialConstraints
import org.apache.spark.sql._
import org.apache.spark.sql.types.StringType

object DepGraphApi extends RepairBase {

  def computeFunctionalDeps(inputView: String, constraintFilePath: String): String = {
    logBasedOnLevel(s"computeFunctionalDep called with: discretizedInputView=$inputView " +
      s"constraintFilePath=$constraintFilePath")

    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName("", inputView)

    var file: Source = null
    val constraints = try {
      file = Source.fromFile(new URI(constraintFilePath).getPath)
      file.getLines()
      DenialConstraints.parseAndVerifyConstraints(file.getLines(), qualifiedName, inputDf.columns)
    } finally {
      if (file != null) {
        file.close()
      }
    }

    // TODO: Reuse the previous computation result
    val domainSizes = {
      // TODO: `StringType` only supported now
      val supported = inputDf.schema.filter(_.dataType == StringType).map(_.name).toSet
      RepairApi.computeAndGetTableStats(inputView).mapValues(_.distinctCount)
        .filter(kv => supported.contains(kv._1))
    }

    val fdMap = mutable.Map[String, mutable.Set[String]]()

    def hasNoCyclic(ref1: String, ref2: String): Boolean = {
      !fdMap.get(ref1).exists(_.contains(ref2)) && !fdMap.get(ref2).exists(_.contains(ref1))
    }

    constraints.predicates.filter { preds =>
      // Filters predicate candidates that might mean functional deps
      preds.length == 2 && preds.flatMap(_.references).distinct.length == 2
    }.foreach {
      case Seq(p1, p2) if Set(p1.sign, p2.sign) == Set("EQ", "IQ") &&
        p1.references.length == 1 && p2.references.length == 1 =>
        val ref1 = p1.references.head
        val ref2 = p2.references.head
        (domainSizes.get(ref1), domainSizes.get(ref2)) match {
          case (Some(ds1), Some(ds2)) if ds1 < ds2 =>
            fdMap.getOrElseUpdate(ref1, mutable.Set[String]()) += ref2
          case (Some(ds1), Some(ds2)) if ds1 > ds2 =>
            fdMap.getOrElseUpdate(ref2, mutable.Set[String]()) += ref1
          case (Some(_), Some(_)) if hasNoCyclic(ref1, ref2) =>
            fdMap(ref1) = mutable.Set(ref2)
          case _ =>
        }
      case _ =>
    }

    // TODO: We need a smarter way to convert Scala data to a json string
    fdMap.map { case (k, values) =>
      s""""$k": [${values.toSeq.sorted.map { v => s""""$v"""" }.mkString(",")}]"""
    }.mkString("{", ",", "}")
  }

  def computeFunctionDepMap(inputView: String, X: String, Y: String): String = {
    val df = spark.sql(
      s"""
         |SELECT CAST($X AS STRING) x, CAST(y[0] AS STRING) y FROM (
         |  SELECT $X, collect_set($Y) y
         |  FROM $inputView
         |  GROUP BY $X
         |  HAVING size(y) = 1
         |)
       """.stripMargin)

    // TODO: We need a smarter way to convert Scala data to a json string
    df.collect.map { case Row(x: String, y: String) =>
      s""""$x": "$y""""
    }.mkString("{", ",", "}")
  }
}
