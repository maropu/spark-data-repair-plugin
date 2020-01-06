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

import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.util.Utils

object IntegrityConstraintDiscovery extends Logging {

  private val DC2FD_CONVERSION_ENABLED = true

  private val BITMASK_EQUAL = 1         // 'mask & BITMASK_EQUAL > 0' means '='; '!=' otherwise
  private val BITMASK_GREATER_THAN = 2  // 'mask & BITMASK_GREATER_THAN > 0' means '>'; '<=' otherwise
  private val BITMASK_LESS_THAN = 4     // 'mask & BITMASK_LESS_THAN > 0' means '<'; '>=' otherwise

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

  private def isCondidateType(dataType: DataType): Boolean = dataType match {
    case IntegerType => true
    case StringType => true
    case _ => false
  }

  def exec(sparkSession: SparkSession, table: String): Map[String, Seq[String]] = {
    withSQLConf(SQLConf.CROSS_JOINS_ENABLED.key -> "true") {
      val fields = sparkSession.table(table).schema.filter { f =>
        isCondidateType(f.dataType)
      }
      if (fields.nonEmpty) {
        val (l, r) = ("leftTable", "rightTable")
        val colNames = fields.map(_.name)
        val (evExprs, whereExprs) = {
          val exprs = colNames.map { colName =>
            val (lc, rc) = (s"$l.$colName", s"$r.$colName")
            def evFunc(cmp: String, pos: Int) = s"SHIFTLEFT(CAST($lc $cmp $rc AS BYTE), $pos)"
            val evs = Seq(evFunc("=", 0), evFunc(">", 1), evFunc("<", 2))
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

          logDebug(evQuery)
          val evVectors = sparkSession.sql(evQuery)

          if (log.isDebugEnabled()) {
            val exprs = colNames.map { colName =>
              def maskFunc(mask: Int) = s"CAST($colName & $mask > 0 AS INT)"
              val masks = Seq(maskFunc(BITMASK_EQUAL), maskFunc(BITMASK_GREATER_THAN),
                maskFunc(BITMASK_LESS_THAN))
              s"(${masks.mkString(" || ")}) AS $colName"
            }
            withTempView(sparkSession, evVectors.selectExpr(exprs :+ "cnt": _*)) { tempView =>
              sparkSession.sql(
                s"""
                   |SELECT
                   |  ${colNames.mkString(", ")}, SUM(cnt) AS cnt
                   |FROM
                   |  $tempView
                   |GROUP BY
                   |  ${colNames.mkString(", ")}
                 """.stripMargin
              ).show(numRows = 100, truncate = false)
            }
          }

          // TODO: Currently, it uses evidences for equality
          val totalCount = evVectors.groupBy().sum("cnt").collect.map { case Row(l: Long) => l }.head
          val equalEv = colNames.map { colName =>
            s"CAST($colName & $BITMASK_EQUAL > 0 AS INT) AS $colName"
          }
          withTempView(sparkSession, evVectors) { tempView =>
            val evDf = sparkSession.sql(
              s"""
                 |SELECT ${colNames.mkString(", ")}, SUM(cnt / $totalCount) AS rt
                 |FROM (
                 |  SELECT
                 |    ${equalEv.mkString(", ")}, cnt
                 |  FROM
                 |    $tempView)
                 |GROUP BY
                 |  ${colNames.mkString(", ")}
               """.stripMargin)

            val localEvidences = evDf.collect()
            val numSymbols = 2
            val rtIdx = fields.length

            fields.indices.combinations(numSymbols).flatMap { indices =>
              val evMap = localEvidences.map { r =>
                indices.map(r.getInt).toArray.toSeq -> r.getDouble(rtIdx)
              }.toMap

              evMap.flatMap { case (evVec, _) =>
                def negateSinglePredicate(pos: Int): Seq[Int] = {
                  // Forcibly copy it
                  val ev = evVec.map(x => x).toArray
                  ev(pos) = ev(pos) ^ 1
                  ev
                }
                // TODO: Currently, it generates DC candidates just by negating
                // a single predicate held in `evVec`.
                val dcCandidates = (0 until numSymbols).map(negateSinglePredicate)
                dcCandidates.flatMap { dcCandidate  =>
                  val violateConstraint = evMap.get(dcCandidate)
                  if (violateConstraint.isEmpty ||
                      violateConstraint.head < sparkSession.sessionState.conf.constraintInferenceApproximateEpilon) {
                    val dcVecWithField = indices.map(fields).zip(dcCandidate).sortBy(_._2)
                    // If `dcVec` has a single '!='(that is, the others are '='),
                    // we can rewrite it to FD.
                    if (sparkSession.sessionState.conf.constraintInferenceDc2fdConversionEnabled &&
                        dcCandidate.count(_ == 0) == 1) {
                      val fieldNames = dcVecWithField.map(_._1.name)
                      val X = fieldNames.init.mkString(",")
                      val Y = fieldNames.last
                      Some(fieldNames.head -> s"FD($X=>$Y)")
                    } else {
                      Some(dcVecWithField.head._1.name -> dcVecWithField.map { case (f, ev) =>
                        s"X.${f.name} ${if (ev > 0) "=" else "!="} Y.${f.name}"
                      }.mkString("DC(", ",", ")"))
                    }
                  } else {
                    None
                  }
                }
              }
            }.toSeq.groupBy(_._1).map {
              case (k, v) => (k, v.map(_._2))
            }
          }
        }
      } else {
        Map.empty
      }
    }
  }
}
