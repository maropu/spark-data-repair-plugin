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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.types._

case class EntityWithRowNum(name: String, numRows: Long, fields: Seq[StructField])

object SchemaOnlyFkInference extends Logging {
  import FkInference._

  private def isValidAttrPair(pair: Seq[(StructField, (String, Long))]): Boolean = pair match {
    case Seq((leftField, (leftTableName, _)), (rightField, (rightTableName, _))) =>
      leftTableName != rightTableName &&
        leftField.dataType.sameType(rightField.dataType)
  }

  def infer(sparkSession: SparkSession, tables: Seq[TableIdentifier]): ResultType = {
    val tableCandidates = tables.flatMap { table =>
      val fields = sparkSession.table(table.identifier).schema.filter { f =>
        isFkCondidateType(f.dataType)
      }
      if (fields.nonEmpty) {
        val t = table.identifier
        Some(EntityWithRowNum(t, sparkSession.table(t).count(), fields))
      } else {
        None
      }
    }

    val fkCandidates = tableCandidates.filter(_.numRows > 0).combinations(2).toSeq
    val fkConstraints = {
      val numTasks = new AtomicInteger(fkCandidates.length)
      val progressBar = new ConsoleProgressBar(numTasks)
      val sparkOutput = new ByteArrayOutputStream()
      val retVal = Console.withErr(sparkOutput) {
        fkCandidates.flatMap { case Seq(lhs, rhs) =>
          logDebug(s"${lhs.name}(${StructType(lhs.fields).toDDL}) <==> " +
            s"${rhs.name}(${StructType(rhs.fields).toDDL})")

          val allAttrs = lhs.fields.zip(lhs.fields.indices.map(_ => (lhs.name, lhs.numRows))) ++
            rhs.fields.zip(rhs.fields.indices.map(_ => (rhs.name, rhs.numRows)))

          val maybeFk = allAttrs.combinations(2).filter(isValidAttrPair).toSeq.sortBy {
            case Seq((leftField, _), (rightField, _)) =>
              computeFkScoreFromName(leftField.name, rightField.name)
          }.collectFirst {
            case Seq((leftField, (leftTableName, leftCounft)), (rightField, (rightTableName, rightCount)))
              if leftField.name == rightField.name =>
              if (leftCounft >= rightCount) {
                leftTableName -> (leftField.name, (rightTableName, rightField.name))
              } else {
                rightTableName -> (rightField.name, (leftTableName, leftField.name))
              }
          }

          numTasks.decrementAndGet()
          maybeFk
        }
      }

      logDebug(sparkOutput.toString)
      progressBar.stop()
      retVal
    }

    fkConstraints.groupBy(_._1).map { case (k, v) =>
      (k, v.map(_._2))
    }
  }
}
