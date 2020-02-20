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

package org.apache.spark.api

import org.apache.commons.lang.RandomStringUtils
import org.apache.spark.SparkException
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.util.Utils

package object python {

  def withSparkSession[T](f: SparkSession => T): T = {
    SparkSession.getActiveSession.map { sparkSession =>
      f(sparkSession)
    }.getOrElse {
      throw new SparkException("An active SparkSession not found.")
    }
  }

  def withTempView[T](spark: SparkSession, df: DataFrame, cache: Boolean = false)(f: String => T): T = {
    val tempView = getRandomString("temp_view_")
    if (cache) df.cache()
    df.createOrReplaceTempView(tempView)
    val ret = f(tempView)
    spark.sql(s"DROP VIEW $tempView")
    ret
  }

  def getRandomString(prefix: String = ""): String = {
    s"$prefix${Utils.getFormattedClassName(this)}_${RandomStringUtils.randomNumeric(12)}"
  }
}
