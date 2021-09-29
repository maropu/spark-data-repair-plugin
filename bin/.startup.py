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
import warnings

from pyspark.sql import SparkSession

# Initializes a Spark session
# NOTE: Since learning tasks run longer, we set a large value (6h)
# to the network timeout value.
spark = SparkSession.builder \
    .config("spark.network.timeout", "21600s") \
    .enableHiveSupport() \
    .getOrCreate()

# Suppress warinig messages in PySpark
warnings.simplefilter('ignore')

# Supresses `WARN` messages in JVM
spark.sparkContext.setLogLevel("ERROR")

# For debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# spark.sql("SET spark.scavenger.logLevel=TRACE")

# Sets Spark configs for data repairing
spark.sql("SET spark.sql.cbo.enabled=true")
spark.sql("SET spark.sql.statistics.histogram.enabled=true")
spark.sql("SET spark.sql.statistics.histogram.numBins=254")
# TODO: Spark v3.1.1 jobs fail because the plan structural integrity is broken
spark.sql("SET spark.sql.optimizer.excludedRules="
          "org.apache.spark.sql.catalyst.optimizer.PropagateEmptyRelation")

# Tunes # shuffle partitions
num_tasks_per_core = 1
num_parallelism = spark.sparkContext.defaultParallelism
spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism * num_tasks_per_core}")

# Defines an entrypoint for Scavenger APIs
from repair.api import Scavenger
scavenger = Scavenger.getOrCreate()

print(f"Scavenger APIs (version {Scavenger.version()}) available as 'scavenger'.")
