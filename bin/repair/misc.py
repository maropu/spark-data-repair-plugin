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

from typing import Any, Optional

from pyspark import SparkContext
from pyspark.sql import DataFrame

from repair.base import ApiBase, ResultBase


class ScavengerRepairMisc(ApiBase):

    def __init__(self) -> None:
        super().__init__()

        self.table_name: Optional[str] = None
        self.row_id: Optional[str] = None
        self.target_attr_list: str = ""
        self.k: Optional[int] = None
        self.q: int = 2
        self.clustering_alg: str = "bisect-kmeans"
        self.null_ratio: float = 0.01

        # JVM interfaces for Scavenger APIs
        self._svg_api = self._jvm.ScavengerMiscApi

    def setTableName(self, table_name: str) -> "ScavengerRepairMisc":
        self.table_name = table_name
        return self

    def setRowId(self, row_id: str) -> "ScavengerRepairMisc":
        self.row_id = row_id
        return self

    def setTargetAttrList(self, target_attr_list: str) -> "ScavengerRepairMisc":
        self.target_attr_list = target_attr_list
        return self

    def setK(self, k: int) -> "ScavengerRepairMisc":
        self.k = k
        return self

    def setQ(self, q: int) -> "ScavengerRepairMisc":
        self.q = q
        return self

    def setClusteringAlg(self, alg: str) -> "ScavengerRepairMisc":
        self.clustering_alg = alg
        return self

    def setNullRatio(self, null_ratio: float) -> "ScavengerRepairMisc":
        self.null_ratio = null_ratio
        return self

    def flatten(self) -> DataFrame:
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before flattening")

        jdf = self._svg_api.flattenTable(self.db_name, self.table_name, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)

    def splitInputTableInto(self) -> DataFrame:
        if self.table_name is None or self.row_id is None or self.k is None:
            raise ValueError("`setTableName`, `setRowId`, and `setK` should be called "
                             "before computing row groups")

        options = f"q={self.q},clusteringAlg={self.clustering_alg}"
        jdf = self._svg_api.splitInputTableInto(
            self.k, self.db_name, self.table_name, self.row_id, self.target_attr_list, options)
        return DataFrame(jdf, self._spark._wrapped)

    def injectNull(self) -> DataFrame:
        if self.table_name is None:
            raise ValueError("`setTableName` should be called before injecting NULL")

        jdf = self._svg_api.injectNullAt(
            self.db_name, self.table_name, self.target_attr_list, self.null_ratio)
        return DataFrame(jdf, self._spark._wrapped)

    def computeAndGetStats(self) -> DataFrame:
        if self.table_name is None:
            raise ValueError("`setTableName` should be called before injecting NULL")

        jdf = self._svg_api.computeAndGetStats(self.db_name, self.table_name)
        return DataFrame(jdf, self._spark._wrapped)

    def toErrorMap(self, error_cells: str) -> DataFrame:
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before flattening")

        jdf = self._svg_api.toErrorMap(error_cells, self.db_name, self.table_name, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)


class SchemaSpy(ApiBase):

    _instance: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SchemaSpy":
        if cls._instance is None:
            cls._instance = super(SchemaSpy, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.driver_name: str = "sqlite"
        self.props: str = ""

        # JVM interfaces for SchemaSpy APIs
        self.spy_api = self._jvm.SchemaSpyApi

    @staticmethod
    def getOrCreate() -> "SchemaSpy":
        return SchemaSpy()

    def setDriverName(self, driver_name: str) -> "SchemaSpy":
        self.driver_name = driver_name
        return self

    def setProps(self, props: str) -> "SchemaSpy":
        self.props = props
        return self

    def catalogToDataFrame(self) -> DataFrame:
        jdf = self.spy_api.catalogToDataFrame(self.db_name, self.driver_name, self.props)
        df = DataFrame(jdf, self._spark._wrapped)
        return df

    def run(self) -> ResultBase:
        result_path = self.spy_api.run(
            self.output, self.db_name, self.driver_name, self.props)
        return ResultBase(result_path)


# This is a method to use SchemaSpy functionality directly
def spySchema(args: str = "") -> None:
    sc = SparkContext._active_spark_contex
    sc._jvm.SchemaSpyApi.run(args)
