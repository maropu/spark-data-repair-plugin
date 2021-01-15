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

sealed trait Expr

case class AttrRef(ident: String) extends Expr {
  override def toString: String = ident
}

case class Constant(value: String) extends Expr {
  override def toString: String = value
}

case class Predicate(genCmp: (String, String) => String, leftExpr: Expr, rightExpr: Expr) {

  def references: Seq[String] =  {
    Seq(leftExpr, rightExpr).filter(_.isInstanceOf[AttrRef]).map(_.toString)
  }

  private def toStringWithQualifier(expr: Expr, qualifier: String): String  = expr match {
    case ref: AttrRef => s"$qualifier.$ref"
    case constant => s"$constant"
  }

  override def toString(): String = {
    val left = toStringWithQualifier(leftExpr, DenialConstraints.leftRelationIdent)
    val right = toStringWithQualifier(rightExpr, DenialConstraints.rightRelationIdent)
    genCmp(left, right)
  }
}

case class DenialConstraints(predicates: Seq[Seq[Predicate]], references: Seq[String])

object DenialConstraints extends Logging {

  val leftRelationIdent = "__generated_left"
  val rightRelationIdent = "__generated_right"

  val emptyConstraints = DenialConstraints(Nil, Nil)

  // TODO: These entries below must be synced with `IntegrityConstraintDiscovery`
  private val opSigns = Seq("EQ", "IQ", "LT", "GT")
  private val signMap: Map[String, (String, String) => String] = Map(
    "EQ" -> ((l: String, r: String) => s"$l <=> $r"),
    "IQ" -> ((l: String, r: String) => s"NOT($l <=> $r)"),
    "LT" -> ((l: String, r: String) => s"$l < $r"),
    "GT" -> ((l: String, r: String) => s"$l > $r"))

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
                Some(Predicate(signMap(cmp), AttrRef(leftAttr), AttrRef(rightAttr)))
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
              case predicate(cmp, leftAttr, value) =>
                Some(Predicate(signMap(cmp), AttrRef(leftAttr), Constant(value)))
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
