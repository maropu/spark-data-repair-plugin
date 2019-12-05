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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.types._

object FkInferType extends Enumeration {
  val NOOP, BASIC, SCHEMA_ONLY = Value
}

object FkInference {

  type ResultType = Map[String, Seq[(String, (String, String))]]

  def isFkCondidateType(dataType: DataType): Boolean = dataType match {
    case IntegerType => true
    case StringType => true
    case _ => false
  }

  def computeFkScoreFromName(X: String, Y: String): Double = {
    val likely = (s: String) => if (Seq("id", "key", "sk").exists(s.contains)) 0.5 else 0.0
    likely(X) + likely(Y)
  }
}

object NoopFkInference {
  def infer(sparkSession: SparkSession, tables: Seq[TableIdentifier])
    : FkInference.ResultType = {
    Map.empty
  }
}

case class FkInference(sparkSession: SparkSession, tpe: FkInferType.Value) {
  import FkInference._

  val doInfer = tpe match {
    case FkInferType.SCHEMA_ONLY => SchemaOnlyFkInference.infer _
    case FkInferType.BASIC => BasicFkInference.infer _
    case FkInferType.NOOP => NoopFkInference.infer _
  }

  def infer(tables: Seq[TableIdentifier]): ResultType = {
    doInfer(sparkSession, tables)
  }
}
