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

import scala.util.control.NonFatal
import scala.util.Try

import org.apache.spark.internal.Logging

sealed trait Expr

case class AttrRef(ident: String) extends Expr {
  override def toString: String = ident
}

case class Constant(value: String) extends Expr {
  override def toString: String = value
}

case class Predicate(
  sign: String,
  genCmp: (String, String) => String,
  leftExpr: Expr,
  rightExpr: Expr) {

  def references: Seq[String] =  {
    Seq(leftExpr, rightExpr).filter(_.isInstanceOf[AttrRef]).map(_.toString).distinct
  }

  private def toStringWithQualifier(expr: Expr, qualifier: String): String = expr match {
    case ref: AttrRef => s"$qualifier.$ref"
    case constant => s"$constant"
  }

  override def toString(): String = {
    val left = toStringWithQualifier(leftExpr, DenialConstraints.leftRelationIdent)
    val right = toStringWithQualifier(rightExpr, DenialConstraints.rightRelationIdent)
    genCmp(left, right)
  }
}

/**
 * This class manages data integrity rules defined by Denial Constraints [5].
 * Syntax definition in this class follows the HoloClean one: http://www.holoclean.io/
 */
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


  def parseAndVerifyConstraints(
      lines: Seq[String],
      inputName: String,
      tableAttrs: Seq[String]): DenialConstraints = {
    val predicates = lines.flatMap { c =>
      try {
        Some(Try(parse(c)).getOrElse(parseAlt(c)))
      } catch {
        case NonFatal(_) =>
          logWarning(s"Illegal constraint format found: $c")
          None
      }
    }

    // Checks if all the attributes contained in `constraintFilePath` exist in `table`
    val refs = predicates.flatMap(_.flatMap(_.references)).distinct
    val constraints = DenialConstraints(predicates.filter(_.nonEmpty), refs)
    val attrsInConstraints = constraints.references
    val tableAttrSet = tableAttrs.toSet
    val absentAttrs = attrsInConstraints.filterNot(tableAttrSet.contains)
    if (absentAttrs.nonEmpty) {
      logWarning(s"Non-existent constraint attributes found in '$inputName': ${absentAttrs.mkString(", ")}")
      val newPredEntries = constraints.predicates.filter { ps =>
        ps.forall { p =>
          p.references.forall(tableAttrSet.contains)
        }
      }
      if (newPredEntries.nonEmpty) {
        constraints.copy(
          predicates = newPredEntries,
          references = constraints.references.filter(tableAttrs.contains))
      } else {
        DenialConstraints.emptyConstraints
      }
    } else {
      constraints
    }
  }

  case class ParseResult(p: Option[Predicate], origText: String = "")

  private def isIdentifier(s: String): Boolean = {
    s.matches("[a-zA-Z]+[a-zA-Z0-9]*")
  }

  // The format like this: "t1&t2&EQ(t1.fk1,t2.fk1)&IQ(t1.v4,t2.v4)"
  private[python] def parse(c: String): Seq[Predicate] = {
    c.split("&").map(_.trim).toSeq match {
      case t1 +: t2 +: constraints if isIdentifier(t1) && isIdentifier(t2) =>
        if (constraints.length >= 2) {
          val predicateDef = s"""(${opSigns.mkString("|")})\\s*\\(\\s*$t1\\.(.*)\\s*,\\s*$t2\\.(.*)\\s*\\)""".r
          val parsed = constraints.map {
            case predicateDef(cmp, leftAttr, rightAttr) =>
              ParseResult(Some(Predicate(cmp, signMap(cmp),
                AttrRef(leftAttr.trim), AttrRef(rightAttr.trim))))
            case s =>
              ParseResult(None, s)
          }
          if (parsed.forall(_.p.isDefined)) {
            val es = parsed.flatMap(_.p)
            logDebug(s"$c => ${es.mkString(" AND ")}")
            es
          } else {
            throw new IllegalArgumentException("Illegal predicates found: " +
              parsed.filterNot(_.p.isDefined).map(_.origText).mkString(", "))
          }
        } else {
          throw new IllegalArgumentException("At least two predicate candidates should be given, " +
            s"but ${constraints.length} candidates found: $c")
        }

      case t1 +: constraints if isIdentifier(t1) =>
        if (constraints.length >= 2) {
          val predicateDef = s"""(${opSigns.mkString("|")})\\s*\\(\\s*$t1\\.(.*)\\s*,\\s*(.*)\\)""".r
          val parsed = constraints.map {
            case predicateDef(cmp, leftAttr, value) =>
              ParseResult(Some(Predicate(cmp, signMap(cmp),
                AttrRef(leftAttr.trim), Constant(value.trim))))
            case s =>
              ParseResult(None, s)
          }
          if (parsed.forall(_.p.isDefined)) {
            val es = parsed.flatMap(_.p)
            logDebug(s"$c => ${es.mkString(",")}")
            es
          } else {
            throw new IllegalArgumentException("Illegal predicates found: " +
              parsed.filterNot(_.p.isDefined).map(_.origText).mkString(", "))
          }
        } else {
          throw new IllegalArgumentException("At least two predicate candidates should be given, " +
            s"but ${constraints.length} candidates found: $c")
        }

      case ps if ps.nonEmpty =>
        throw new IllegalArgumentException(s"Failed to parse an input string: '$c'")

      case _ =>
        Nil
    }
  }

  // The format like this: "X->Y;Y->Z"
  private[python] def parseAlt(c: String): Seq[Predicate] = {
    c.split("->").map(_.trim()).filter(_.nonEmpty).toSeq match {
      case Seq(x, y) =>
        Seq(Predicate("EQ", signMap("EQ"), AttrRef(x), AttrRef(x)),
          Predicate("IQ", signMap("IQ"), AttrRef(y), AttrRef(y)))
      case ps if ps.nonEmpty =>
        throw new IllegalArgumentException(s"Failed to parse an input string: '$c'")
      case _ =>
        Nil
    }
  }
}
