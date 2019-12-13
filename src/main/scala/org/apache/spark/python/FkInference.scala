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

  // TODO: We might be able to use an attribute name to infer if the attribute is a foreign key or not.
  // There is the earlier study to use attribute name strings for type inferences:
  //  * Vraj Shah and Arun Kumar, The ML Data Prep Zoo: Towards Semi-Automatic Data Preparation for ML,
  //    Proceedings of DEEM'19, Article 11, 2019, https://doi.org/10.1145/3329486.3329499.
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

  //  TODO: Needs to implement more sophisticated FK detection algorithms:
  //   * Kruse, Sebastian, et al., Fast Approximate Discovery of Inclusion Dependencies,
  //     Proceedings of BTW'17, 2017, https://dl.gi.de/handle/20.500.12116/629
  val doInfer = tpe match {
    case FkInferType.SCHEMA_ONLY => SchemaOnlyFkInference.infer _
    case FkInferType.BASIC => BasicFkInference.infer _
    case FkInferType.NOOP => NoopFkInference.infer _
  }

  def infer(tables: Seq[TableIdentifier]): ResultType = {
    doInfer(sparkSession, tables)
  }
}
