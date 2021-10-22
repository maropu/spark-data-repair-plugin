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

import java.util.Locale

import scala.language.implicitConversions

import org.apache.spark.internal.config.{ConfigBuilder, ConfigEntry, ConfigReader}
import org.apache.spark.sql.internal.SQLConf

object RepairConf {

  /**
   * Implicitly injects the [[RepairConf]] into [[SQLConf]].
   */
  implicit def SQLConfToRepairConf(conf: SQLConf): RepairConf = new RepairConf(conf)

  private val sqlConfEntries = java.util.Collections.synchronizedMap(
    new java.util.HashMap[String, ConfigEntry[_]]())

  private def register(entry: ConfigEntry[_]): Unit = sqlConfEntries.synchronized {
    require(!sqlConfEntries.containsKey(entry.key),
      s"Duplicate SQLConfigEntry. ${entry.key} has been registered")
    sqlConfEntries.put(entry.key, entry)
  }

  def buildConf(key: String): ConfigBuilder = ConfigBuilder(key).onCreate(register)

  val LOG_LEVEL = buildConf("spark.repair.logLevel")
    .internal()
    .doc("Configures the logging level. The value can be 'trace', 'debug', 'info', 'warn', or 'error'. " +
      "The default log level is 'trace'.")
    .stringConf
    .transform(_.toUpperCase(Locale.ROOT))
    .checkValue(logLevel => Set("TRACE", "DEBUG", "INFO", "WARN", "ERROR").contains(logLevel),
      "Invalid value for 'spark.repair.logLevel'. Valid values are 'trace', " +
        "'debug', 'info', 'warn' and 'error'.")
    .createWithDefault("trace")
}

class RepairConf(conf: SQLConf) {
  import RepairConf._

  private val reader = new ConfigReader(conf.settings)

  def logLevel: String = getConf(LOG_LEVEL)

  /**
   * Return the value of configuration property for the given key. If the key is not set yet,
   * return `defaultValue` in [[ConfigEntry]].
   */
  private def getConf[T](entry: ConfigEntry[T]): T = {
    require(sqlConfEntries.get(entry.key) == entry || SQLConf.isStaticConfigKey(entry.key),
      s"$entry is not registered")
    entry.readFrom(reader)
  }
}
