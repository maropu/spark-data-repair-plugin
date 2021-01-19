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

from abc import ABCMeta, abstractmethod
from typing import Optional

from pyspark.sql import DataFrame, SparkSession


class ErrorDetector(metaclass=ABCMeta):

    def __init__(self, name: str) -> None:
        self.name = name
        self.row_id: Optional[str] = None
        self.input_table: Optional[str] = None

        # For Spark/JVM interactions
        self._spark = SparkSession.builder.getOrCreate()
        self._api = self._spark.sparkContext._active_spark_context._jvm.ScavengerErrorDetectorApi

    def setup(self, row_id: str, input_table: str) -> None:
        self.row_id = row_id
        self.input_table = input_table

    @abstractmethod
    def _detect_impl(self) -> DataFrame:
        pass

    def detect(self) -> DataFrame:
        assert self.row_id is not None and self.input_table is not None
        dirty_df = self._detect_impl()
        assert isinstance(dirty_df, DataFrame)
        return dirty_df


class NullErrorDetector(ErrorDetector):

    def __init__(self) -> None:
        ErrorDetector.__init__(self, 'NullErrorDetector')

    def _detect_impl(self) -> DataFrame:
        jdf = self._api.detectNullCells('', self.input_table, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)


class RegExErrorDetector(ErrorDetector):

    def __init__(self, error_pattern: str, error_cells_as_string: bool = False) -> None:
        ErrorDetector.__init__(self, 'RegExErrorDetector')
        self.error_pattern = error_pattern
        self.error_cells_as_string = error_cells_as_string

    def _detect_impl(self) -> DataFrame:
        jdf = self._api.detectErrorCellsFromRegEx(
            '', self.input_table, self.row_id, self.error_pattern,
            self.error_cells_as_string)
        return DataFrame(jdf, self._spark._wrapped)


class ConstraintErrorDetector(ErrorDetector):

    def __init__(self, constraint_path: str) -> None:
        ErrorDetector.__init__(self, 'ConstraintErrorDetector')
        self.constraint_path = constraint_path

    def _detect_impl(self) -> DataFrame:
        jdf = self._api.detectErrorCellsFromConstraints(
            '', self.input_table, self.row_id, self.constraint_path)
        return DataFrame(jdf, self._spark._wrapped)


class OutlierErrorDetector(ErrorDetector):

    def __init__(self, approx_enabled: bool) -> None:
        ErrorDetector.__init__(self, 'OutlierErrorDetector')
        self.approx_enabled = approx_enabled

    def _detect_impl(self) -> DataFrame:
        jdf = self._api.detectErrorCellsFromOutliers(
            '', self.input_table, self.row_id, self.approx_enabled)
        return DataFrame(jdf, self._spark._wrapped)
