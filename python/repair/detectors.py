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
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession  # type: ignore


class ErrorDetector(metaclass=ABCMeta):

    def __init__(self, name: str) -> None:
        self.name = name
        self.row_id: Optional[str] = None
        self.qualified_input_name: Optional[str] = None
        self.targets: List[str] = []

        # For Spark/JVM interactions
        self._spark = SparkSession.builder.getOrCreate()
        self._detector_api = self._spark.sparkContext._active_spark_context._jvm.ErrorDetectorApi  # type: ignore

    def setUp(self, row_id: str, qualified_input_name: str, targets: List[str]) -> "ErrorDetector":
        self.row_id = row_id
        self.qualified_input_name = qualified_input_name
        self.targets = targets
        return self

    @abstractmethod
    def _detect_impl(self) -> DataFrame:
        pass

    def _to_target_list(self) -> str:
        return ','.join(self.targets) if self.targets else ''

    def detect(self) -> DataFrame:
        assert self.row_id is not None and self.qualified_input_name is not None
        dirty_df = self._detect_impl()
        assert isinstance(dirty_df, DataFrame)
        return dirty_df


class NullErrorDetector(ErrorDetector):

    def __init__(self) -> None:
        ErrorDetector.__init__(self, 'NullErrorDetector')

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectNullCells(self.qualified_input_name, self.row_id, self._to_target_list())
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class DomainValues(ErrorDetector):

    def __init__(self, attr: str, values: List[str]) -> None:
        ErrorDetector.__init__(self, 'DomainValues')
        self.attr = attr
        self.domain_values = values

    def _detect_impl(self) -> DataFrame:
        regex = '({})'.format('|'.join(self.domain_values)) if self.domain_values else '$^'
        jdf = self._detector_api.detectErrorCellsFromRegEx(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.attr, regex)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class RegExErrorDetector(ErrorDetector):

    def __init__(self, attr: str, regex: str) -> None:
        ErrorDetector.__init__(self, 'RegExErrorDetector')
        self.attr = attr
        self.regex = regex

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromRegEx(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.attr, self.regex)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class ConstraintErrorDetector(ErrorDetector):

    def __init__(self, constraint_path: str) -> None:
        ErrorDetector.__init__(self, 'ConstraintErrorDetector')
        self.constraint_path = constraint_path

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromConstraints(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.constraint_path)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class OutlierErrorDetector(ErrorDetector):

    def __init__(self, approx_enabled: bool = False) -> None:
        ErrorDetector.__init__(self, 'OutlierErrorDetector')
        self.approx_enabled = approx_enabled

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromOutliers(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.approx_enabled)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore
