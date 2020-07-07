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
import org.apache.spark.internal.Logging

case class Predicate(
    cmp: String,
    leftTable: Option[String],
    leftAttr: String,
    rightTable: Option[String],
    rightAttr: String) {

  def references: Seq[String] = {
    val left = if (leftTable.isDefined) leftAttr :: Nil else Nil
    val right = if (rightTable.isDefined) rightAttr :: Nil else Nil
    left ++ right
  }

  override def toString(): String = {
    val left = leftTable.map(t => s"$t.$leftAttr").getOrElse(leftAttr)
    val right = rightTable.map(t => s"$t.$rightAttr").getOrElse(rightAttr)
    s"$left $cmp $right"
  }
}

object Predicate {

  def apply(c: String, lt: String, la: String, rt: String, ra: String): Predicate = {
    new Predicate(c, Some(lt), la, Some(rt), ra)
  }

  def apply(c: String, lt: String, la: String, constant: String): Predicate = {
    new Predicate(c, Some(lt), la, None, constant)
  }
}

case class DenialConstraints(predicates: Seq[Seq[Predicate]], references: Seq[String]) {

  lazy val leftTable: String = {
    predicates.flatten.flatMap(_.leftTable).distinct.ensuring(_.size < 2)
      .headOption.getOrElse("__auto_generated_1")
  }

  lazy val rightTable: String = {
    predicates.flatten.flatMap(_.rightTable).distinct.ensuring(_.size < 2)
      .headOption.getOrElse("__auto_generated_2")
  }

  def toWhereCondition(predicates: Seq[Predicate]): String = {
    predicates.map(_.toString).mkString(" AND ")
  }
}

object DenialConstraints extends Logging {

  lazy val emptyConstraints = DenialConstraints(Nil, Nil)

  // TODO: These entries below must be synced with `IntegrityConstraintDiscovery`
  private val opSigns = Seq("EQ", "IQ", "LT", "GT")
  private val signMap: Map[String, String] =
    Map("EQ" -> "=", "IQ" -> "!=", "LT" -> "<", "GT" -> ">")

  // The format like this: "t1&t2&EQ(t1.fk1,t2.fk1)&IQ(t1.v4,t2.v4)"
  def parse(path: String): DenialConstraints = {
    var file: Source = null
    try {
      file = Source.fromFile(new URI(path).getPath)
      val isIdentifier = (s: String) => s.matches("[a-zA-Z0-9]+")
      val predicates = mutable.ArrayBuffer[Seq[Predicate]]()
      file.getLines().foreach { dcStr => dcStr.split("&").map(_.trim).toSeq match {
          case t1 +: t2 +: constraints if isIdentifier(t1) && isIdentifier(t2) =>
            val predicate = s"""(${opSigns.mkString("|")})\\($t1\\.(.*),$t2\\.(.*)\\)""".r
            val es = constraints.flatMap {
              case predicate(cmp, leftAttr, rightAttr) =>
                Some(Predicate(signMap(cmp), Some(t1), leftAttr, Some(t2), rightAttr))
              case s =>
                logWarning(s"Illegal predicate format found: $s")
                None
            }
            if (es.nonEmpty) {
              logDebug(s"$dcStr => ${es.mkString(" AND ")}")
              predicates.append(es)
            }
          case t1 +: constraints if isIdentifier(t1) =>
            val predicate = s"""(${opSigns.mkString("|")})\\($t1\\.(.*),(.*)\\)""".r
            val es = constraints.flatMap {
              case predicate(cmp, leftAttr, constant) =>
                Some(Predicate(signMap(cmp), Some(t1), leftAttr, None, constant))
              case s =>
                logWarning(s"Illegal predicate format found: $s")
                None
            }
            if (es.nonEmpty) {
              logDebug(s"$dcStr => ${es.mkString(",")}")
              predicates.append(es)
            }
          case Nil => // Just ignores this case
          case Seq(s) if s.trim.isEmpty =>
          case s => logWarning(s"Illegal constraint format found: ${s.mkString(",")}")
        }
      }

      if (predicates.nonEmpty) {
        val references = predicates.flatMap { _.flatMap(_.references) }.distinct
        DenialConstraints(predicates, references)
      } else {
        logWarning(s"Valid predicate entries not found in `$path`")
        emptyConstraints
      }
    } finally {
      if (file != null) {
        file.close()
      }
    }
  }
}
