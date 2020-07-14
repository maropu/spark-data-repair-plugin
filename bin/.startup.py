#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging

from pyspark.sql import SparkSession

from repair.api import *

# Initializes a Spark session
if not sc._jvm.SparkSession.getActiveSession().isDefined():
    spark.sql("SELECT 1")

# Suppress warinig messages in PySpark
warnings.simplefilter('ignore')

# Supresses `WARN` messages in JVM
spark.sparkContext.setLogLevel("ERROR")

# For debugging
spark.sql("SET spark.scavenger.logLevel=TRACE")
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARN)

# Since 3.0, `spark.sql.crossJoin.enabled` is set to true by default
spark.sql("SET spark.sql.crossJoin.enabled=true")
spark.sql("SET spark.sql.cbo.enabled=true")

# Tunes # shuffle partitions
num_tasks_per_core = 1
num_parallelism = spark.sparkContext.defaultParallelism
spark.sql("SET spark.sql.shuffle.partitions=%s" % (num_parallelism * num_tasks_per_core))

# Defines an entrypoint for Scavenger APIs
scavenger = Scavenger.getOrCreate()

print("Scavenger APIs (version %s) available as 'scavenger'." % ("0.1.0-spark3.0-EXPERIMENTAL"))

