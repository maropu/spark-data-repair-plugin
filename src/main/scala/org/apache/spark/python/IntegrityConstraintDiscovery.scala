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

import org.apache.commons.lang.RandomStringUtils

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.util.Utils

object IntegrityConstraintDiscovery extends Logging {

  val tableConstraintsKey = "__TABLE_CONSTRAINTS_KEY__"

  private val MAX_TARGET_FIELD_NUM = 100
  private val NUM_INTEGERESTINGNESS_SYMBOLS = 2
  private val METADATA_EQUAL = ("EQ", "=", 1)
  private val METADATA_NOT_EQUAL = ("IQ", "!=", 0)
  private val METADATA_GREATER_THAN = ("GT", ">", 2)
  private val METADATA_LESS_THAN = ("LT", "<", 4)
  private val METADATA_PREDICATES = Seq(METADATA_EQUAL, METADATA_GREATER_THAN, METADATA_LESS_THAN)

  private def negateSymbol(s: String) = s match {
    case "EQ" => "IQ"
    case "IQ" => "EQ"
    case "GT" => "LTE"
    case "LT" => "GTE"
  }

  private def withSQLConf[T](pairs: (String, String)*)(f: => T): T= {
    val conf = SQLConf.get
    val (keys, values) = pairs.unzip
    val currentValues = keys.map { key =>
      if (conf.contains(key)) {
        Some(conf.getConfString(key))
      } else {
        None
      }
    }
    (keys, values).zipped.foreach { (k, v) =>
      assert(!SQLConf.staticConfKeys.contains(k))
      conf.setConfString(k, v)
    }
    try f finally {
      keys.zip(currentValues).foreach {
        case (key, Some(value)) => conf.setConfString(key, value)
        case (key, None) => conf.unsetConf(key)
      }
    }
  }

  private def withTempView[T](spark: SparkSession, df: DataFrame)(f: String => T): T = {
    val tempView = getRandomString("tempView_")
    df.createOrReplaceTempView(tempView)
    val ret = f(tempView)
    spark.sql(s"DROP VIEW $tempView")
    ret
  }

  private def getRandomString(prefix: String = ""): String = {
    s"$prefix${Utils.getFormattedClassName(this)}_${RandomStringUtils.randomNumeric(12)}"
  }

  private def catalogStats(sparkSession: SparkSession, table: String): Seq[(String, String)] = {
    val df = sparkSession.table(table)
    val tableNode = df.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
    tableNode.computeStats().attributeStats.map { case (a, stats) =>
      val statStrings = Seq(
        stats.distinctCount.map(v => s"distinctCnt=$v"),
        // stats.min.map(v => s"min=$v"),
        // stats.max.map(v => s"max=$v"),
        stats.nullCount.map(v => s"nullCnt=$v")
        // stats.avgLen.map(v => s"avgLen=$v"),
        // stats.maxLen.map(v => s"maxLen=$v"),
      ).flatten
      a.name -> s"STATS(${statStrings.mkString(",")})"
    }.toSeq
  }

  private def checkConstraints(sparkSession: SparkSession, table: String): Seq[(String, String)] = {
    val fields = sparkSession.table(table).schema.filter { f =>
      f.dataType match {
        case IntegerType => true
        case FloatType | DoubleType => true
        case _ => false
      }
    }
    if (fields.nonEmpty) {
      val tableStats = {
        val df = sparkSession.table(table)
        val tableNode = df.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
        tableNode.computeStats()
      }
      val minMaxStatMap = tableStats.attributeStats.map {
        kv => (kv._1.name, (kv._2.min, kv._2.max))
      }
      val minMaxAggExprs = fields.map(_.name).flatMap { columnName =>
        def aggFunc(f: String, c: String) = s"CAST($f($c) AS STRING)"
        val minFuncExpr = aggFunc("MIN", columnName)
        val maxFuncExpr = aggFunc("MAX", columnName)
        minMaxStatMap.get(columnName).map { case (minOpt, maxOpt) =>
          val minExpr = minOpt.map(v => s"STRING($v)").getOrElse(minFuncExpr)
          val maxExpr = maxOpt.map(v => s"STRING($v)").getOrElse(maxFuncExpr)
          minExpr :: maxExpr :: Nil
        }.getOrElse {
          minFuncExpr :: maxFuncExpr :: Nil
        }
      }

      val queryToComputeStats =
        s"""
           |SELECT ${minMaxAggExprs.mkString(", ")}
           |FROM $table
         """.stripMargin

      logDebug(queryToComputeStats)

      val minMaxStats = sparkSession.sql(queryToComputeStats).take(1).head
      fields.zipWithIndex.map { case (f, i) =>
        val minValue = minMaxStats.getString(2 * i)
        val maxValue = minMaxStats.getString(2 * i + 1)
        (f.name, s"CHK($minValue,$maxValue)")
      }
    } else {
      Seq.empty
    }
  }

  private def isValidMetadata(md: (StructField, (String, String, Int))): Boolean = md match {
    case (StructField(_, StringType, _, _), METADATA_EQUAL | METADATA_NOT_EQUAL) => true
    case (StructField(_, _: NumericType, _, _), _) => true
    case _ => false
  }

  private def denialConstraints(sparkSession: SparkSession, table: String): Seq[(String, String)] = {
    withSQLConf(SQLConf.CROSS_JOINS_ENABLED.key -> "true") {
      val fields = sparkSession.table(table).schema.filter { f =>
        f.dataType match {
          case IntegerType => true
          case StringType => true
          case _ => false
        }
      }
      if (fields.nonEmpty) {
        if (fields.length > MAX_TARGET_FIELD_NUM) {
          throw new SparkException(s"Maximum field length is $MAX_TARGET_FIELD_NUM, " +
            s"but ${fields.length} found.")
        }
        val (l, r) = ("leftTable", "rightTable")
        val colNames = fields.map(_.name)
        val (evExprs, whereExprs) = {
          val exprs = colNames.map { colName =>
            val (lc, rc) = (s"$l.$colName", s"$r.$colName")
            val evs = METADATA_PREDICATES.zipWithIndex.map { case ((_, cmp, _), pos) =>
              s"SHIFTLEFT(CAST($lc $cmp $rc AS BYTE), $pos)"
            }
            (s"(${evs.mkString(" | ")}) AS $colName",
              s"$lc != $rc")
          }
          (exprs.map(_._1).mkString(",\n"), exprs.map(_._2).mkString(" OR\n"))
        }
        val tableRowCount = sparkSession.table(table).count()
        val sampleRatio = {
          val samplingSize = sparkSession.sessionState.conf.samplingSize
          Math.min(samplingSize.toDouble / tableRowCount, 1.0)
        }
        val sampleTableDf = sparkSession.table(table).sample(sampleRatio)
        withTempView(sparkSession, sampleTableDf) { inputView =>
          val evQuery =
            s"""
               |SELECT
               |  ${colNames.mkString(", ")}, COUNT(1) AS cnt
               |FROM (
               |  SELECT
               |    $evExprs
               |  FROM
               |    /* table row count: $tableRowCount */
               |    /* sampling ratio: $sampleRatio */
               |    $inputView $l,
               |    $inputView $r
               |  WHERE
               |    $whereExprs
               |)
               |GROUP BY
               |  ${colNames.mkString(", ")}
             """.stripMargin

          val evVectors = sparkSession.sql(evQuery)
          val totalEvCount = evVectors.groupBy().sum("cnt").collect.map { case Row(l: Long) => l }.head
          logDebug(
            s"""
               |Number of evidences: $totalEvCount
               |Number of aggregated evidences: ${evVectors.count}
               |Query to compute evidences:
               |$evQuery
             """.stripMargin)

          if (log.isDebugEnabled()) {
            val exprs = colNames.map { colName =>
              val masks = METADATA_PREDICATES.map { md => s"CAST($colName & ${md._3} > 0 AS INT)" }
              s"(${masks.mkString(" || ")}) AS $colName"
            }
            withTempView(sparkSession, evVectors.selectExpr(exprs :+ "cnt": _*)) { tempView =>
              // Aggregated evidence bitmap
              sparkSession.sql(
                s"""
                   |SELECT
                   |  ${colNames.mkString(", ")}, SUM(cnt) AS rt
                   |FROM
                   |  $tempView
                   |GROUP BY
                   |  ${colNames.mkString(", ")}
                 """.stripMargin
              ).show(numRows = 300, truncate = false)
            }
          }

          // TODO: Currently, it uses evidences for equality
          val equalEv = colNames.map { colName =>
            s"CAST($colName & ${METADATA_EQUAL._3} > 0 AS INT) AS $colName"
          }
          withTempView(sparkSession, evVectors) { tempView =>
            val numSymbols = NUM_INTEGERESTINGNESS_SYMBOLS
            val evDf = sparkSession.sql(
              s"""
                 |SELECT ${colNames.mkString(", ")}, SUM(cnt) AS cnt
                 |FROM (
                 |  SELECT
                 |    ${equalEv.mkString(", ")}, cnt
                 |  FROM
                 |    $tempView)
                 |GROUP BY
                 |  ${colNames.mkString(", ")}
               """.stripMargin)

            // XXX
            val approxEpsilon = sparkSession.sessionState.conf.constraintInferenceApproximateEpilon
            val threshold = totalEvCount * (1.0 - approxEpsilon)
            val queryToValidateConstraints = fields.combinations(numSymbols).flatMap { fs =>
              (METADATA_NOT_EQUAL +: METADATA_PREDICATES).combinations(numSymbols).flatMap { metadataSeq =>
                val exprs = fs.zip(metadataSeq).filter(isValidMetadata)
                if (exprs.length == numSymbols) {
                  // The format of denial constraints refers to the HoloClean one:
                  //  - https://github.com/HoloClean/holoclean/blob/master/testdata/hospital_constraints.txt
                  val aliasName = exprs.map(v => s"${negateSymbol(v._2._1)}(${v._1.name})").mkString("&amp;")
                  val exprToValidateConstraint = exprs.map(v => s"${v._1.name} = ${v._2._3}").mkString(" OR ")
                  // XXX
                  Some((metadataSeq.map(_._1).zip(fs.map(_.name)),
                    s"SUM(IF($exprToValidateConstraint, cnt, 0)) >= " +
                    s"$threshold AS `$aliasName`"))
                } else {
                  None
                }
              }
            }.toSeq

            // Computes minimal covers of the evidence set for searching denial constraints
            val cDf = evDf.selectExpr(queryToValidateConstraints.map(_._2): _*)
            // Sets false at NULL cells
            val cRow = cDf.na.fill(false).collect().head
            val metadataSeq = queryToValidateConstraints.map(_._1)
            cDf.schema.zipWithIndex.flatMap { case (f, i) =>
              if (cRow.getBoolean(i)) {
                Seq(tableConstraintsKey -> s"X&amp;Y&amp;${f.name}") ++
                  (if (sparkSession.sessionState.conf.constraintInferenceDc2fdConversionEnabled) {
                    metadataSeq(i) match {
                      case Seq(("EQ", x), ("IQ", y)) => y -> s"FD($y->$x)" :: Nil
                      case Seq(("IQ", x), ("EQ", y)) => x -> s"FD($x->$y)" :: Nil
                      case _ => Nil
                    }
                  } else {
                    Nil
                  })
              } else {
                Nil
              }
            }
          }
        }
      } else {
        Seq.empty
      }
    }
  }

  def exec(sparkSession: SparkSession, table: String): Map[String, Seq[String]] = {
    if (sparkSession.table(table).schema.map(_.name).contains(tableConstraintsKey)) {
      throw new SparkException(s"$tableConstraintsKey cannot be used as a column name.")
    }
    val constraints = denialConstraints(sparkSession, table) ++ checkConstraints(sparkSession, table) ++
      catalogStats(sparkSession, table)
    constraints.groupBy(_._1).map { case (k, v) =>
      (k, v.map(_._2))
    }
  }
}
