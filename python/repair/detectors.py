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

    def __init__(self) -> None:
        self.row_id: Optional[str] = None
        self.qualified_input_name: Optional[str] = None
        self.targets: List[str] = []

        # For Spark/JVM interactions
        self._spark = SparkSession.builder.getOrCreate()
        self._detector_api = self._spark.sparkContext._active_spark_context._jvm.ErrorDetectorApi  # type: ignore

    def setUp(self, row_id: str, qualified_input_name: str,
              continous_cols: List[str], targets: List[str]) -> "ErrorDetector":
        self.row_id = row_id
        self.qualified_input_name = qualified_input_name
        self.continous_cols = continous_cols
        self.targets = targets
        return self

    @abstractmethod
    def _detect_impl(self) -> DataFrame:
        pass

    def _to_continous_col_list(self) -> str:
        return ','.join(self.continous_cols) if self.continous_cols else ''

    def _to_target_list(self) -> str:
        return ','.join(self.targets) if self.targets else ''

    def detect(self) -> DataFrame:
        assert self.row_id is not None and self.qualified_input_name is not None
        dirty_df = self._detect_impl()
        assert isinstance(dirty_df, DataFrame)
        return dirty_df


class NullErrorDetector(ErrorDetector):

    def __init__(self) -> None:
        ErrorDetector.__init__(self)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectNullCells(self.qualified_input_name, self.row_id, self._to_target_list())
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class DomainValues(ErrorDetector):

    def __init__(self, attr: str, values: List[str] = [], autofill: bool = False, min_count_thres: int = 12) -> None:
        ErrorDetector.__init__(self)
        self.attr = attr
        self.values = values if not autofill else []
        self.autofill = autofill
        self.min_count_thres = min_count_thres

    def __str__(self) -> str:
        args = f'attr="{self.attr}",size={len(self.values)},autofill={self.autofill},' \
            f'min_count_thres={self.min_count_thres}'
        return f'{self.__class__.__name__}({args})'

    def _detect_impl(self) -> DataFrame:
        domain_values = self.values
        if self.autofill:
            domain_value_df = self._spark.table(str(self.qualified_input_name)) \
                .groupBy(self.attr).count() \
                .where(f'{self.attr} IS NOT NULL AND count > {self.min_count_thres}') \
                .selectExpr(self.attr)

            if domain_value_df.count() > 0:
                domain_values = [r[0] for r in domain_value_df.collect()]

        regex = '({})'.format('|'.join(domain_values)) if domain_values else '$^'
        jdf = self._detector_api.detectErrorCellsFromRegEx(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.attr, regex)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class RegExErrorDetector(ErrorDetector):

    def __init__(self, attr: str, regex: str) -> None:
        ErrorDetector.__init__(self)
        self.attr = attr
        self.regex = regex

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(pattern="{self.regex}")'

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromRegEx(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.attr, self.regex)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class ConstraintErrorDetector(ErrorDetector):

    def __init__(self, constraint_path: str) -> None:
        ErrorDetector.__init__(self)
        self.constraint_path = constraint_path

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(path={self.constraint_path})'

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromConstraints(
            self.qualified_input_name, self.row_id, self._to_target_list(), self.constraint_path)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore


class OutlierErrorDetector(ErrorDetector):

    def __init__(self, approx_enabled: bool = False) -> None:
        ErrorDetector.__init__(self)
        self.approx_enabled = approx_enabled

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(approx_enabled={self.approx_enabled})'

    def _detect_impl(self) -> DataFrame:
        jdf = self._detector_api.detectErrorCellsFromOutliers(
            self.qualified_input_name, self.row_id, self._to_continous_col_list(),
            self._to_target_list(), self.approx_enabled)
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore
