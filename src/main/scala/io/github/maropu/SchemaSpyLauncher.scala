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

package io.github.maropu

import scala.collection.JavaConverters._
import scala.io.Source

import org.apache.spark.internal.Logging

import io.github.maropu.Utils._

object SchemaSpyLauncher extends Logging {

  private val SCHEMASPY_PACKAGE = "SCHEMASPY_PACKAGE_PATH"

  def run(arguments: String*): Unit = {
    logDebug(s"SchemaSpy arguments: ${arguments.mkString(" ")}")

    val schemaSpyPath = Utils.getEnvOrFail(SCHEMASPY_PACKAGE)
    try {
      val cmd = Seq("java", "-jar", schemaSpyPath) ++ arguments
      val builder = new ProcessBuilder(cmd.asJava)
      val proc = builder.start()
      outputToConsole(s"\n${Source.fromInputStream(proc.getInputStream).mkString}")
      proc.waitFor()
    } catch {
      case e => System.err.println(e.getMessage)
    }
  }
}
