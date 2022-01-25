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

from pyspark.sql import DataFrame, SparkSession  # type: ignore[import]

from repair.utils import argtype_check


class RepairMisc():
    """Interface to provide helper functionalities.

    .. versionchanged:: 0.1.0
    """

    def __init__(self) -> None:
        super().__init__()

        self.opts: Dict[str, str] = {}

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._misc_api_ = self._spark.sparkContext._active_spark_context._jvm.RepairMiscApi  # type: ignore

    @argtype_check
    def option(self, key: str, value: str) -> "RepairMisc":
        """Adds an input option for misc functions.

        .. versionchanged:: 0.1.0
        """
        self.opts[str(key)] = str(value)
        return self

    @argtype_check
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

    @property
    def _num_bins(self) -> int:
        if "num_bins" in self.opts.keys():
            return int(self.opts["num_bins"])
        else:
            return 8

    def _parse_option(self, key: str, default: str) -> str:
        if key in self.opts.keys():
            return self.opts[key]
        else:
            return default

    def _check_required_options(self, required: List[str]) -> None:
        if not all(opt in self.opts.keys() for opt in required):
            raise ValueError("Required options not found: {}".format(", ".join(required)))

    def repair(self) -> DataFrame:
        """Applies predicted repair updates into an input table.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> spark.table("predicted").show(3)
        +---+---------+-----------+
        |tid|attribute|   repaired|
        +---+---------+-----------+
        |  5|      Age|      18-21|
        |  5|   Income|MoreThan50K|
        |  7|      Sex|     Female|
        | 12|      Age|      18-21|
        | 12|      Sex|     Female|
        |  3|      Sex|     Female|
        | 16|   Income|MoreThan50K|
        +---+---------+-----------+

        >>> df = delphi.misc.options({"repair_updates": "predicted", "table_name": "adult",
        ...    "row_id": "tid"}).repair()
        >>> df.where("tid in ('3', '5', '7', '12', '16')").show()
        +---+-----+------------+---------------+------------+------+-------------+-----------+
        |tid|  Age|   Education|     Occupation|Relationship|   Sex|      Country|     Income|
        +---+-----+------------+---------------+------------+------+-------------+-----------+
        | 16|  >50|Some-college|Exec-managerial|   Unmarried|Female|United-States|MoreThan50K|
        |  5|18-21|Some-college|   Craft-repair|     Husband|  Male|United-States|MoreThan50K|
        | 12|18-21|   Bachelors|Exec-managerial|     Husband|Female|United-States|MoreThan50K|
        |  3|22-30|     HS-grad|   Craft-repair|   Own-child|Female|United-States|LessThan50K|
        |  7|31-50| Prof-school| Prof-specialty|     Husband|Female|        India|MoreThan50K|
        +---+-----+------------+---------------+------------+------+-------------+-----------+
        """
        self._check_required_options(["repair_updates", "table_name", "row_id"])
        jdf = self._misc_api_.repairAttrsFrom(
            self.opts["repair_updates"],
            self._db_name,
            self.opts["table_name"],
            self.opts["row_id"])
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def describe(self) -> DataFrame:
        """Computes column stats for an input table.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = delphi.misc.option("table_name", "adult").describe()
        >>> df.show()
        +------------+-----------+----+----+-------+------+------+----+
        |    attrName|distinctCnt| min| max|nullCnt|avgLen|maxLen|hist|
        +------------+-----------+----+----+-------+------+------+----+
        |     Country|          3|null|null|      0|    13|    13|null|
        |   Education|          7|null|null|      0|     9|    12|null|
        |      Income|          2|null|null|      2|    11|    11|null|
        |  Occupation|          7|null|null|      0|    13|    17|null|
        |Relationship|          4|null|null|      0|     9|    13|null|
        |         tid|         20|null|null|      0|     2|     2|null|
        |         Sex|          2|null|null|      3|     5|     6|null|
        |         Age|          4|null|null|      2|     5|     5|null|
        +------------+-----------+----+----+-------+------+------+----+
        """
        self._check_required_options(["table_name"])
        jdf = self._misc_api_.computeAndGetStats(
            self._db_name, self.opts["table_name"], self._num_bins)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def flatten(self) -> DataFrame:
        """Converts an input table into a flatten <row_id, attribute, vaue> table.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = delphi.misc.options({"table_name": "adult", "row_id": "tid"}).flatten()
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
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def splitInputTable(self) -> DataFrame:
        """Splits an input table into multiple tables with similar rows.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = delphi.misc.options({"table_name": "adult", "row_id": "tid", "k": "2"})
        ...    .splitInputTable()
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

        param_q = self._parse_option("q", "2")
        param_alg = self._parse_option("clustering_alg", "bisect-kmeans")
        param_options = f"q={param_q},clusteringAlg={param_alg}"

        jdf = self._misc_api_.splitInputTableInto(
            int(self.opts["k"]), self._db_name, self.opts["table_name"], self.opts["row_id"],
            self._target_attr_list, param_options)

        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def injectNull(self) -> DataFrame:
        """Randomly injects NULL into given attributes.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> spark.range(10).write.saveAsTable("t")
        >>> df = delphi.misc.options({"table_name": "t", "target_attr_list": "id",
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
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def _show_histogram(self, df: DataFrame) -> None:
        import matplotlib.pyplot as plt  # type: ignore[import]
        fig = plt.figure()
        num_targets = df.count()
        for index, row in enumerate(df.collect()):
            pdf = df.where(f'attribute = "{row.attribute}"').selectExpr("inline(histogram)").toPandas()
            print(pdf)
            f = fig.add_subplot(num_targets, 1, index + 1)
            f.bar(pdf["value"], pdf["cnt"])
            f.set_xlabel(row.attribute)
            f.set_ylabel("cnt")

        fig.tight_layout()
        fig.show()

    def toHistogram(self) -> DataFrame:
        """Computes the histogram of values in specified targets.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = delphi.misc.options({"table_name": "adult", "row_id": "tid",
        ...    "targets": "Income,Age,Relationship,Sex"}).toHistogram()
        >>> df.show()
        +------------+------------------------------------------------------------------+
        |attribute   |histogram                                                         |
        +------------+------------------------------------------------------------------+
        |Income      |[{MoreThan50K, 1}, {LessThan50K, 12}]                             |
        |Age         |[{>50, 3}, {22-30, 2}, {31-50, 7}, {18-21, 1}]                    |
        |Relationship|[{Not-in-family, 3}, {Unmarried, 1}, {Husband, 6}, {Own-child, 3}]|
        |Sex         |[{Female, 4}, {Male, 9}]                                          |
        +------------+------------------------------------------------------------------+
        """
        self._check_required_options(["table_name", "targets"])
        jdf = self._misc_api_.convertToHistogram(self.opts["targets"], self._db_name, self.opts["table_name"])
        hist_df = DataFrame(jdf, self._spark._wrapped)  # type: ignore
        # self._show_histogram(hist_df)
        return hist_df

    def toErrorMap(self) -> DataFrame:
        """Converts an input table into an error map.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> df = delphi.misc.options({"table_name": "adult", "row_id": "tid",
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
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def generateDepGraph(self) -> None:
        """Generates a dependency graph for a specified input table.

        .. versionchanged:: 0.1.0

        Examples
        --------
        >>> delphi.misc.options({"table_name": "adult", "path": "/tmp/adult",
        ...    "pairwise_attr_stat_threshold": "1.0"}).generateDepGraph()
        """
        self._check_required_options(["path", "table_name"])
        param_max_domain_size = self._parse_option("max_domain_size", "100")
        param_max_attr_value_num = self._parse_option("max_attr_value_num", "30")
        param_max_attr_value_length = self._parse_option("max_attr_value_length", "70")
        param_pairwise_attr_stat_threshold = self._parse_option("pairwise_attr_stat_threshold", "1.0")
        param_edge_label = len(self._parse_option("edge_label", "")) > 0
        param_filename_prefix = self._parse_option("filename_prefix", "depgraph")
        param_overwrite = len(self._parse_option("overwrite", "")) > 0
        jdf = self._misc_api_.generateDepGraph(
            self.opts["path"], self._db_name, self.opts["table_name"],
            "svg", self._target_attr_list, int(param_max_domain_size),
            int(param_max_attr_value_num), int(param_max_attr_value_length),
            float(param_pairwise_attr_stat_threshold), param_edge_label,
            param_filename_prefix, param_overwrite)
