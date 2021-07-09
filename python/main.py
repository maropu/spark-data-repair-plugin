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

import datetime
import logging
import warnings
from argparse import ArgumentParser

from pyspark.sql import SparkSession


def _create_temp_name(prefix="temp"):
    return f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'


if __name__ == "__main__":
    # Parses command-line arguments
    parser = ArgumentParser()
    # TODO: Support more options for building a repair model
    parser.add_argument('--db', dest='db', help='Database Name', type=str, required=False, default='default')
    parser.add_argument('--input', dest='input', help='Input Table Name', type=str, required=True)
    parser.add_argument('--row-id', dest='row_id', help='Unique Row ID', type=str, required=True)
    parser.add_argument('--output', dest='output', help='Output Table Name', type=str, required=True)
    args = parser.parse_args()

    # Initializes a Spark session
    # NOTE: Since learning tasks run longer, we set a large value (6h)
    # to the network timeout value.
    spark = SparkSession.builder \
        .config("spark.network.timeout", "21600s") \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppress warinig messages in Python
    warnings.simplefilter('ignore')

    # Supresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    # For debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
    repaired_df = scavenger.repair \
        .setTableName(args.input) \
        .setRowId(args.row_id) \
        .run()

    try:
        repaired_df.write.saveAsTable(args.output)
    except:
        temp_output_table_name = _create_temp_name()
        repaired_df.write.saveAsTable(temp_output_table_name)
        print(f"Table '{args.output}' already exists, so saved the predicted repair values "
              f"as '{temp_output_table_name}' instead")
    else:
        print(f"Predicted repair values are saved as '{args.output}'")
    finally:
        spark.stop()
