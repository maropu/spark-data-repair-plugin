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

import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicInteger

import org.apache.commons.lang.RandomStringUtils
import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types._

import io.github.maropu.Utils._

case class EntityWithStats(name: String, numRows: Long, columnStats: Seq[(StructField, Long)])

object BasicFkInference extends Logging {
  import org.apache.spark.python.FkInference._

  private def isValidAttrPair(pair: Seq[((StructField, Long), (String, Long))]): Boolean = pair match {
    case Seq(((leftField, leftDistinctCount), leftTable),
        ((rightField, rightDistinctCount), rightTable)) =>
      leftTable != rightTable &&
        leftField.dataType.sameType(rightField.dataType) &&
        leftDistinctCount == rightDistinctCount
  }

  def infer(sparkSession: SparkSession, tables: Seq[TableIdentifier]): ResultType = {
    val tableCandidates = tables.flatMap { table =>
      val fields = sparkSession.table(table.identifier).schema.filter { f =>
        isFkCondidateType(f.dataType)
      }
      if (fields.nonEmpty) {
        Some((table.identifier, fields))
      } else {
        None
      }
    }

    outputConsole("Collecting basic stats in input tables...")

    // Generates a sequence of table pairs for FK discoveries
    val entityWithStats = {
      val numTasks = new AtomicInteger(tableCandidates.length)
      val progressBar = new ConsoleProgressBar(numTasks)
      val sparkOutput = new ByteArrayOutputStream()
      val retVal = Console.withErr(sparkOutput) {
        tableCandidates.map { case (tableName, fields) =>
          // Cache tables in advance
          sparkSession.sql(s"CACHE TABLE $tableName")

          val queryToComputeStats = {
            val tableStats = {
              val df = sparkSession.table(tableName)
              val tableNode = df.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
              tableNode.computeStats()
            }
            val attrStatMap = tableStats.attributeStats.map {
              kv => (kv._1.name, kv._2.distinctCount)
            }
            // If we have stats in catalog, we just use them
            val rowCount = tableStats.rowCount.map { cnt => s"bigint(${cnt.toLong}) /* rowCount */" }
              .getOrElse("COUNT(1)")
            val distinctCounts = fields.map { f =>
              val aggFunc = s"COUNT(DISTINCT ${f.name})"
              attrStatMap.get(f.name).map {
                distinctCntOpt => distinctCntOpt.map { v => s"bigint(${v.toLong}) /* ${f.name} */" }
                  .getOrElse(aggFunc)
              }.getOrElse(aggFunc)
            }
            s"""
               |SELECT
               |  $rowCount,
               |  ${distinctCounts.mkString(",\n  ")}
               |FROM
               |  $tableName
             """.stripMargin
          }

          logDebug(s"Query to compute $tableName stats:" + queryToComputeStats)

          val statsRow = sparkSession.sql(queryToComputeStats).take(1).head
          val es = EntityWithStats(tableName, statsRow.getLong(0), fields.zipWithIndex.map {
            case (f, i) => (f, statsRow.getLong(i + 1))
          })
          numTasks.decrementAndGet()
          es
        }
      }

      logDebug(sparkOutput.toString)
      progressBar.stop()
      retVal
    }

    logDebug({
      val statsStr = entityWithStats.map { e =>
        val columns = e.columnStats.map { case (f, distinctCount) =>
          s"|${f.name}(${f.dataType.catalogString})|=$distinctCount"
        }
        s"|${e.name}|=${e.numRows}: ${columns.mkString(", ")}"
      }
      s"\n${statsStr.mkString("\n")}"
    })

    outputConsole("Analyzing FK constraints for the them...")

    val fkCandidates = entityWithStats.filter(_.numRows > 0).combinations(2).toSeq
    val inferredFkConstraints = {
      val numTasks = new AtomicInteger(fkCandidates.length)
      val progressBar = new ConsoleProgressBar(numTasks)
      val sparkOutput = new ByteArrayOutputStream()
      val retVal = Console.withErr(sparkOutput) {
        object FkConstraint {
          def unapply(pair: Seq[((StructField, Long), (String, Long))])
          : Option[(String, (String, (String, String)))] = pair match {
            case Seq(((leftField, leftDistinctCount), (leftTableName, leftCount)),
              ((rightField, rightDistinctCount), (rightTableName, rightCount)))
                if leftDistinctCount == rightDistinctCount =>

              val (largerTable, smallerTable) = if (leftCount >= rightCount) {
                ((leftTableName, leftField.name, leftCount),
                  (rightTableName, rightField.name, rightCount))
              } else {
                ((rightTableName, rightField.name, rightCount),
                  (leftTableName, leftField.name, leftCount))
              }

              val largerTempView = s"largerTempView_${RandomStringUtils.randomNumeric(12)}"
              val sampleRatio = {
                val fkInferenceSamplingSize = sparkSession.sessionState.conf.fkInferenceSamplingSize
                100.0 * Math.min(fkInferenceSamplingSize.toDouble / largerTable._3, 1.0)
              }
              sparkSession.sql(
                s"""
                   |CACHE TABLE $largerTempView AS
                   |  SELECT *
                   |    FROM ${largerTable._1}
                   |    TABLESAMPLE ($sampleRatio PERCENT)
                 """.stripMargin)

              val largerTempViewCount = sparkSession.table(largerTempView).count
              val resultDf = sparkSession.sql(
                s"""
                   |SELECT COUNT(1) = $largerTempViewCount
                   |  FROM $largerTempView l, ${smallerTable._1} r
                   |  WHERE l.${largerTable._2} = r.${smallerTable._2}
               """.stripMargin)

              sparkSession.sql(s"DROP VIEW $largerTempView")

              val mayHaveFkConstraint = resultDf.take(1).map {
                case Row(r: Boolean) => r
              }.head
              if (mayHaveFkConstraint) {
                Some(largerTable._1 -> (largerTable._2, (smallerTable._1, smallerTable._2)))
              } else {
                None
              }
            case _ =>
              None
          }
        }

        fkCandidates.flatMap { case Seq(lhs, rhs) =>
          logDebug(s"${lhs.name}(${StructType(lhs.columnStats.map(_._1)).toDDL}) <==> " +
            s"${rhs.name}(${StructType(rhs.columnStats.map(_._1)).toDDL})")

          val allAttrs = lhs.columnStats.zip(lhs.columnStats.indices.map(_ => (lhs.name, lhs.numRows))) ++
            rhs.columnStats.zip(rhs.columnStats.indices.map(_ => (rhs.name, rhs.numRows)))

          val candidates = allAttrs.combinations(2).filter(isValidAttrPair).toSeq.sortBy {
            case Seq(((leftField, _), _), ((rightField, _), _)) =>
              // To sort attribute pairs in a descending order, computes `1.0 - score`
              1.0 - computeFkScoreFromName(leftField.name, rightField.name)
          }

          logDebug({
            val fkCandidatesWithScores = candidates.map {
                case Seq(((leftField, _), (leftTable, _)), ((rightField, _), (rightTable, _))) =>
              val score = computeFkScoreFromName(leftField.name, rightField.name)
              s" - score($leftTable.${leftField.name}, $rightTable.${rightField.name})=$score"
            }
            s"""
               |A FK candidates list for ${lhs.name}/${rhs.name}:
               |${fkCandidatesWithScores.mkString("\n")}
             """.stripMargin
          })

          val maybeFk = candidates.collectFirst { case FkConstraint(a, b) => a -> b }
          numTasks.decrementAndGet()
          maybeFk
        }
      }

      logDebug(sparkOutput.toString)
      progressBar.stop()
      retVal
    }

    // Uncaches all the tables
    tableCandidates.foreach { case (tableName, _) =>
      sparkSession.sql(s"UNCACHE TABLE $tableName")
    }

    inferredFkConstraints.groupBy(_._1).map { case (k, v) =>
      (k, v.map(_._2))
    }
  }
}
