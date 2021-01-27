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
    """Interface to provide helper functionalities.

    .. versionchanged:: 0.1.0
    """

    def __init__(self) -> None:
        super().__init__()

        self.opts: Dict[str, str] = {}

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._misc_api_ = self._spark.sparkContext._active_spark_context._jvm.RepairMiscApi

    def option(self, key: str, value: str) -> "RepairMisc":
        """Adds an input option for misc functions.

        .. versionchanged:: 0.1.0
        """
        self.opts[str(key)] = str(value)
        return self

    def options(self, options: Dict[str, str]) -> "RepairMisc":
        """Adds input options for misc functions.

        .. versionchanged:: 0.1.0
        """
        self.opts.update(options)
        return self

    @property
    def _db_name(self) -> str:
        if "db_name" in self.opts.keys():
            return self.opts["db_name"]
        else:
            return ""

    @property
    def _target_attr_list(self) -> str:
        if "target_attr_list" in self.opts.keys():
            return self.opts["target_attr_list"]
        else:
            return ""

    def _check_required_options(self, required: List[str]) -> None:
        if not all(opt in self.opts.keys() for opt in required):
            raise ValueError("Required options not found: {}".format(", ".join(required)))

    def flatten(self) -> DataFrame:
        """Converts an input table into a flatten <row_id, attribute, vaue> table.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = scavenger.misc.options({"table_name": "adult", "row_id": "tid"}).flatten()
        >>> df.show(3)
        +---+------------+---------------+
        |tid|   attribute|          value|
        +---+------------+---------------+
        |  0|         Age|          31-50|
        |  0|   Education|   Some-college|
        |  0|  Occupation|   Craft-repair|
        +---+------------+---------------+
        only showing top 3 rows
        """
        self._check_required_options(["table_name", "row_id"])
        jdf = self._misc_api_.flattenTable(
            self._db_name, self.opts["table_name"], self.opts["row_id"])
        return DataFrame(jdf, self._spark._wrapped)

    def splitInputTableInto(self) -> DataFrame:
        """Splits an input table into multiple tables with similar rows.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = scavenger.misc.options({"table_name": "adult", "row_id": "tid", "k": "2"})
        ...    .splitInputTableInto()
        >>> df.show(3)
        +---+---+
        |tid|  k|
        +---+---+
        |  0|  0|
        |  1|  0|
        |  2|  1|
        +---+---+
        only showing top 3 rows
        """
        self._check_required_options(["table_name", "row_id", "k"])

        if not self.opts["k"].isdigit():
            raise ValueError(f"Option 'k' must be an integer, but '{self.opts['k']}' found")

        param_q = self.opts["q"] if "q" in self.opts.keys() else "2"
        param_alg = self.opts["clustering_alg"] if "clustering_alg" in self.opts.keys() \
            else "bisect-kmeans"
        param_options = f"q={param_q},clusteringAlg={param_alg}"

        jdf = self._misc_api_.splitInputTableInto(
            int(self.opts["k"]), self._db_name, self.opts["table_name"], self.opts["row_id"],
            self._target_attr_list, param_options)

        return DataFrame(jdf, self._spark._wrapped)

    def injectNull(self) -> DataFrame:
        """Randomly injects NULL into given attributes.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> spark.range(10).write.saveAsTable("t")
        >>> df = scavenger.misc.options({"table_name": "t", "target_attr_list": "id",
        ...    "null_ratio": "0.9"}).injectNull()
        >>> df.show()
        +----+
        |  id|
        +----+
        |null|
        |null|
        |null|
        |null|
        |null|
        |null|
        |   0|
        |null|
        |null|
        |null|
        +----+
        """
        self._check_required_options(["table_name", "target_attr_list"])

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

        jdf = self._misc_api_.injectNullAt(
            self._db_name, self.opts["table_name"], self._target_attr_list,
            param_null_ratio)
        return DataFrame(jdf, self._spark._wrapped)

    def computeAndGetStats(self) -> DataFrame:
        """Computes column stats for an input table.

        .. versionchanged:: 0.1.0

        """
        self._check_required_options(["table_name"])
        jdf = self._misc_api_.computeAndGetStats(self._db_name, self.opts["table_name"])
        return DataFrame(jdf, self._spark._wrapped)

    def toErrorMap(self) -> DataFrame:
        """Converts an input table into an error map.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = scavenger.misc.options({"table_name": "adult", "row_id": "tid",
        ...    "error_cells": "error_cells"}).toErrorMap()
        >>> df.show()
        +---+---------+
        |tid|error_map|
        +---+---------+
        |  0|  -------|
        |  1|  -------|
        |  2|  -------|
        |  3|  ----*--|
        |  4|  -------|
        |  5|  *-----*|
        |  6|  -------|
        |  7|  ----*--|
        |  8|  -------|
        |  9|  -------|
        | 10|  -------|
        | 11|  -------|
        | 12|  *---*--|
        | 13|  -------|
        | 14|  -------|
        | 15|  -------|
        | 16|  ------*|
        | 17|  -------|
        | 18|  -------|
        | 19|  -------|
        +---+---------+

        """
        self._check_required_options(["table_name", "row_id", "error_cells"])
        jdf = self._misc_api_.toErrorMap(
            self.opts["error_cells"], self._db_name, self.opts["table_name"],
            self.opts["row_id"])
        return DataFrame(jdf, self._spark._wrapped)
