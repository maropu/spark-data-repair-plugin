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

from typing import Dict, List

from pyspark.sql import DataFrame, SparkSession


class RepairMisc():

    def __init__(self) -> None:
        super().__init__()

        self.opts: Dict[str, str] = {}

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._svg_api = self._spark.sparkContext._active_spark_context._jvm.RepairMiscApi

    def options(self, options: Dict[str, str]) -> "RepairMisc":
        """
        Adds input options for the misc functions.
        """
        self.opts.update(options)
        return self

    def _db_name(self) -> str:
        if "db_name" in self.opts.keys():
            return self.opts["db_name"]
        else:
            return ""

    def _target_attr_list(self) -> str:
        if "target_attr_list" in self.opts.keys():
            return self.opts["target_attr_list"]
        else:
            return ""

    def _check_required_options(self, required: List[str]) -> None:
        if not all(opt in self.opts.keys() for opt in required):
            raise ValueError("Required options not found: {}".format(", ".join(required)))

    def flatten(self) -> DataFrame:
        self._check_required_options(["table_name", "row_id"])
        jdf = self._svg_api.flattenTable(
            self._db_name(), self.opts["table_name"], self.opts["row_id"])
        return DataFrame(jdf, self._spark._wrapped)

    def splitInputTableInto(self) -> DataFrame:
        self._check_required_options(["table_name", "row_id", "k"])

        if not self.opts["k"].isdigit():
            raise ValueError(f"Option 'k' must be an integer, but '{self.opts['k']}' found")

        param_q = self.opts["q"] if "q" in self.opts.keys() else "2"
        param_alg = self.opts["clustering_alg"] if "clustering_alg" in self.opts.keys() \
            else "bisect-kmeans"
        param_options = f"q={param_q},clusteringAlg={param_alg}"

        jdf = self._svg_api.splitInputTableInto(
            int(self.opts["k"]), self._db_name(), self.opts["table_name"], self.opts["row_id"],
            self._target_attr_list(), param_options)

        return DataFrame(jdf, self._spark._wrapped)

    def injectNull(self) -> DataFrame:
        self._check_required_options(["table_name"])

        if "null_ratio" in self.opts.keys():
            try:
                param_null_ratio = float(self.opts["null_ratio"])
                is_float = True
            except ValueError:
                is_float = False
            if not (is_float and 0.0 < param_null_ratio <= 1.0):
                raise ValueError("Option 'null_ratio' must be a float in (0.0, 1.0], "
                                 f"but '{self.opts['null_ratio']}' found")
        else:
            param_null_ratio = 0.01

        jdf = self._svg_api.injectNullAt(
            self._db_name(), self.opts["table_name"], self._target_attr_list(),
            param_null_ratio)
        return DataFrame(jdf, self._spark._wrapped)

    def computeAndGetStats(self) -> DataFrame:
        self._check_required_options(["table_name"])
        jdf = self._svg_api.computeAndGetStats(self._db_name(), self.opts["table_name"])
        return DataFrame(jdf, self._spark._wrapped)

    def toErrorMap(self) -> DataFrame:
        self._check_required_options(["table_name", "row_id", "error_cells"])
        jdf = self._svg_api.toErrorMap(
            self.opts["error_cells"], self._db_name(), self.opts["table_name"],
            self.opts["row_id"])
        return DataFrame(jdf, self._spark._wrapped)
