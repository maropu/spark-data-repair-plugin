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

import scala.collection.JavaConverters._

import org.antlr.v4.runtime.tree.TerminalNode
import org.antlr.v4.runtime.{BaseErrorListener, CharStreams, CommonTokenStream, RecognitionException, Recognizer}

import org.apache.spark.SparkException

object RegexTokenType extends Enumeration {
  val Pattern, Constant, Other = Value
}

private[python] class RegexRewriteVisiter
    extends RegexBaseBaseVisitor[Seq[(RegexTokenType.Value, String)]] {
  import RegexBaseParser._

  private def parse(n: TerminalNode, tpe: RegexTokenType.Value) = {
    if (n != null) (tpe, n.toString) :: Nil else Nil
  }

  override def visitRegularExpression(ctx: RegularExpressionContext)
    : Seq[(RegexTokenType.Value, String)] = {
    parse(ctx.CARET, RegexTokenType.Other) ++
      visitExpression(ctx.expression) ++
      parse(ctx.DOLLAR, RegexTokenType.Other)
  }

  override def visitExpression(ctx: ExpressionContext)
    : Seq[(RegexTokenType.Value, String)] = {
    if (ctx.expression().size > 0) {
      ctx.expression().asScala.flatMap(visitExpression)
    } else if (ctx.CONSTANT() != null) {
      parse(ctx.CONSTANT, RegexTokenType.Constant)
    } else if (ctx.RANGE() != null) {
      parse(ctx.RANGE, RegexTokenType.Pattern)
    } else {
      Nil
    }
  }
}

private[python] case object ParseErrorListener extends BaseErrorListener {

  override def syntaxError(
      recognizer: Recognizer[_, _],
      offendingSymbol: scala.Any,
      line: Int,
      charPositionInLine: Int,
      msg: String,
      e: RecognitionException): Unit = {
    throw new SparkException(msg)
  }
}

private[python] object RegexParser {

  def parse(pattern: String): Seq[(RegexTokenType.Value, String)] = {
    val lexer = new RegexBaseLexer(CharStreams.fromString(pattern))
    lexer.addErrorListener(ParseErrorListener)

    val tokens = new CommonTokenStream(lexer)
    val parser = new RegexBaseParser(tokens)
    val visiter = new RegexRewriteVisiter()
    visiter.visit(parser.regularExpression())
  }
}

/**
 * Extracts a cell value structure from a specified regular expression and
 * repair dirty cell values using the structural information.
 *
 * TODO: We currently use a naive approach to extract the structure, but
 * more sophisticated ways have been proposed recently, e.g.,
 * - Zeyu Li et al., Repairing data through regular expressions,
 *   Proceedings of the VLDB Endowment, vol.9, no.5, pp.432-443, 2016.
 */
case class RegexStructureRepair(pattern: String) {

  private val (repairRegex, repairFunc, numPatterns) = {
    val tokenSeq = RegexParser.parse(pattern)
    val numPatterns = tokenSeq.count(_._1 == RegexTokenType.Pattern)
    val regex = tokenSeq.map { case (tpe, token) => tpe match {
      case RegexTokenType.Pattern => s"($token)"
      case RegexTokenType.Constant => s".{1,${token.length}}"
      case RegexTokenType.Other => token
    }}.mkString.r

    val repair = (data: Seq[String]) => {
      tokenSeq.foldLeft(("", 0)) { case ((s, index), (tpe, token)) => tpe match {
        case RegexTokenType.Pattern => (s"$s${data(index)}", index + 1)
        case RegexTokenType.Constant => (s"$s$token", index)
        case RegexTokenType.Other => (s, index)
      }}._1
    }

    (regex, repair, numPatterns)
  }

  def apply(s: String): Option[String] = {
    if (s != null) {
      repairRegex.findFirstMatchIn(s).map { matched =>
        assert(matched.groupCount == numPatterns, s"Illegal pattern found: $pattern")
        repairFunc(matched.subgroups)
      }
    } else {
      None
    }
  }
}