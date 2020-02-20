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

object ScavengerConf {

  /**
   * Implicitly injects the [[ScavengerConf]] into [[SQLConf]].
   */
  implicit def SQLConfToScavengerConf(conf: SQLConf): ScavengerConf = new ScavengerConf(conf)

  private val sqlConfEntries = java.util.Collections.synchronizedMap(
    new java.util.HashMap[String, ConfigEntry[_]]())

  private def register(entry: ConfigEntry[_]): Unit = sqlConfEntries.synchronized {
    require(!sqlConfEntries.containsKey(entry.key),
      s"Duplicate SQLConfigEntry. ${entry.key} has been registered")
    sqlConfEntries.put(entry.key, entry)
  }

  def buildConf(key: String): ConfigBuilder = ConfigBuilder(key).onCreate(register)

  val SAMPLING_SIZE = buildConf("spark.scavenger.samplingSize")
    .internal()
    .doc("Sampling size to infer FK constraints in catalog tables.")
    .intConf
    .createWithDefault(1000)

  val FK_INFERENCE_APPROX_COUNT_ENABLED = buildConf("spark.scavenger.fkInference.approxCount.enabled")
    .internal()
    .doc("Whether to use approximate distinct counting for FK constraint inference.")
    .booleanConf
    .createWithDefault(false)

  val CONSTRAINT_INFERENCE_APPROXIMATE_EPSILON =
    buildConf("spark.scavenger.constraintInference.approxEpsilon")
      .doc("Epsilon value for approximate constraint inferences")
      .doubleConf
      .checkValue(v => 0.0 <= v && v < 1.0, "The epsilon value must be in [0.0, 1.0].")
      .createWithDefault(0.01)

  val CONSTRAINT_INFERENCE_TOPK =
    buildConf("spark.scavenger.constraintInference.topK")
      .doc("Whether to only show topK entries for integrity constraint with high confidences. " +
        "By setting this value to 0, this feature can be disabled.")
      .intConf
      .checkValue(v => v >= 0, "The topK value must not be negative.")
      .createWithDefault(0)

  val CONSTRAINT_INFERENCE_DC2FD_CONVERSION_ENABLED =
    buildConf("spark.scavenger.constraintInference.dc2fdConversion.enabled")
      .internal()
      .doc("Whether to transform denial constraints into functional dependencies if possible.")
      .booleanConf
      .createWithDefault(true)

  val LOG_LEVEL = buildConf("spark.scavenger.logLevel")
    .internal()
    .doc("Configures the logging level. The value can be 'trace', 'debug', 'info', 'warn', or 'error'. " +
      "The default log level is 'trace'.")
    .stringConf
    .transform(_.toUpperCase(Locale.ROOT))
    .checkValue(logLevel => Set("TRACE", "DEBUG", "INFO", "WARN", "ERROR").contains(logLevel),
      "Invalid value for 'spark.scavenger.logLevel'. Valid values are 'trace', " +
        "'debug', 'info', 'warn' and 'error'.")
    .createWithDefault("trace")
}

class ScavengerConf(conf: SQLConf) {
  import ScavengerConf._

  private val reader = new ConfigReader(conf.settings)

  def samplingSize: Int = getConf(SAMPLING_SIZE)
  def fkInferenceApproxCountEnabled: Boolean = getConf(FK_INFERENCE_APPROX_COUNT_ENABLED)
  def constraintInferenceApproximateEpilon: Double = getConf(CONSTRAINT_INFERENCE_APPROXIMATE_EPSILON)
  def constraintInferenceTopK: Int = getConf(CONSTRAINT_INFERENCE_TOPK)
  def constraintInferenceDc2fdConversionEnabled: Boolean = getConf(CONSTRAINT_INFERENCE_DC2FD_CONVERSION_ENABLED)

  def logLevel: String = getConf(LOG_LEVEL)

  /**
   * Return the value of configuration property for the given key. If the key is not set yet,
   * return `defaultValue` in [[ConfigEntry]].
   */
  private def getConf[T](entry: ConfigEntry[T]): T = {
    require(sqlConfEntries.get(entry.key) == entry || SQLConf.staticConfKeys.contains(entry.key),
      s"$entry is not registered")
    entry.readFrom(reader)
  }
}
