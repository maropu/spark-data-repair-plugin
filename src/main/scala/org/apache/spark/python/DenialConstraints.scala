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

import java.net.URI

import scala.collection.mutable
import scala.io.Source
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging

case class Predicate(cmp: String, leftAttr: String, rightAttr: String) {
  // TODO: Currently, comparisons on the same attributes are supported only
  assert(leftAttr == rightAttr)
}

case class DenialConstraints(entries: Seq[Seq[Predicate]], attrNames: Seq[String])

object DenialConstraints extends Logging {

  // TODO: These entries below must be synced with `IntegrityConstraintDiscovery`
  private val opSigns = Seq("EQ", "IQ", "LT", "GT")
  private val signMap: Map[String, String] =
    Map("EQ" -> "=", "IQ" -> "!=", "LT" -> "<", "GT" -> ">")

  def toWhereCondition(predicates: Seq[Predicate], left: String, right: String): String = {
    predicates.map { p =>
      s"$left.${p.leftAttr} ${p.cmp} $right.${p.rightAttr}"
    }.mkString(" AND ")
  }

  // The format like this: "t1&t2&EQ(t1.fk1,t2.fk1)&IQ(t1.v4,t2.v4)"
  def parse(path: String): DenialConstraints = {
    var file: Source = null
    try {
      file = Source.fromFile(new URI(path).getPath)
      val predicates = mutable.ArrayBuffer[Seq[Predicate]]()
      file.getLines().foreach { dcStr => dcStr.split("&").toSeq match {
          case t1 +: t2 +: constraints =>
            val predicate = s"""(${opSigns.mkString("|")})\\($t1\\.(.*),$t2\\.(.*)\\)""".r
            val es = constraints.flatMap {
              case predicate(cmp, leftAttr, rightAttr) =>
                Some(Predicate(signMap(cmp), leftAttr, rightAttr))
              case s =>
                logWarning(s"Illegal predicate format found: $s")
                None
            }
            if (es.nonEmpty) {
              logWarning(s"$dcStr => ${toWhereCondition(es, t1, t2)}")
              predicates.append(es)
            }
          case s =>
            logWarning(s"Illegal constraint format found: $s")
        }
      }
      if (predicates.isEmpty) {
        throw new SparkException(s"Valid predicate entries not found in `$path`")
      }
      val attrNames = predicates.flatMap { _.flatMap { p => p.leftAttr :: p.rightAttr :: Nil }}.distinct
      DenialConstraints(predicates, attrNames)
    } finally {
      if (file != null) {
        file.close()
      }
    }
  }
}
