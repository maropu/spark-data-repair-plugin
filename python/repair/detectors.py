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
import functools
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

from pyspark.sql import DataFrame, SparkSession, functions  # type: ignore
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


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

    def _emptyDataFrame(self) -> DataFrame:
        input_schema = self._spark.table(str(self.qualified_input_name)).schema
        row_id_field = input_schema[str(self.row_id)]
        schema = StructType([row_id_field, StructField("attribute", StringType())])
        return self._spark.createDataFrame(self._spark.sparkContext.emptyRDD(), schema)

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
        if self.attr in self.continous_cols:
            return self._emptyDataFrame()

        domain_values = self.values
        if self.autofill:
            domain_value_df = self._spark.table(str(self.qualified_input_name)) \
                .groupBy(self.attr).count() \
                .where(f'{self.attr} IS NOT NULL AND count > {self.min_count_thres}') \
                .selectExpr(self.attr)

            if domain_value_df.count() > 0:
                domain_values = [str(r[0]) for r in domain_value_df.collect()]

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


class GaussianOutlierErrorDetector(ErrorDetector):

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


class ScikitLearnBasedErrorDetector(ErrorDetector):

    def __init__(self, parallel_mode_threshold: int = 10000) -> None:
        ErrorDetector.__init__(self)
        self.parallel_mode_threshold = parallel_mode_threshold

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    # NOTE: XXX
    @abstractmethod
    def _outlier_detector_impl(self) -> Any:
        pass

    def _create_temp_name(self, prefix: str) -> str:
        return f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    def _detect_impl(self) -> DataFrame:
        columns = list(filter(lambda c: c in self.targets, self.continous_cols)) if self.targets \
            else self.continous_cols
        if not columns:
            return self._emptyDataFrame()

        outlier_detectors = {c: self._outlier_detector_impl() for c in columns}
        input_df = self._spark.table(str(self.qualified_input_name))
        num_rows = input_df.count()

        if num_rows < self.parallel_mode_threshold:
            pdf = input_df.toPandas()
            row_ids = pdf[self.row_id]

            sdfs: List[DataFrame] = []
            for c in columns:
                # Since we assume a specified outlier detector cannot handle NaN cells,
                # we fill them with median values before detecting errors..
                median = np.median(pdf[[c]].dropna())
                predicted = outlier_detectors[c].fit_predict(pdf[[c]].fillna(median))
                error_cells = row_ids[predicted < 0]
                if len(error_cells) > 0:
                    sdf = self._spark.createDataFrame(pd.DataFrame(error_cells, columns=[self.row_id]))
                    sdfs.append(sdf.selectExpr(f'{self.row_id}', f'"{c}" attribute'))

            return functools.reduce(lambda x, y: x.union(y), sdfs) if sdfs \
                else self._emptyDataFrame()

        predicted_fields = [StructField(c, IntegerType()) for c in columns]
        output_schema = StructType([input_df.schema[str(self.row_id)]] + predicted_fields)

        broadcasted_row_id = self._spark.sparkContext.broadcast(str(self.row_id))
        broadcasted_columns = self._spark.sparkContext.broadcast(columns)
        broadcasted_clfs = self._spark.sparkContext.broadcast(outlier_detectors)

        @functions.pandas_udf(output_schema, functions.PandasUDFType.GROUPED_MAP)
        def predict(pdf: pd.DataFrame) -> pd.DataFrame:
            row_id = broadcasted_row_id.value
            columns = broadcasted_columns.value
            clfs = broadcasted_clfs.value

            _pdf = pdf[[row_id]]
            for c in columns:
                # Since we assume a specified outlier detector cannot handle NaN cells,
                # we fill them with median values before detecting errors..
                median = np.median(pdf[[c]].dropna())
                _pdf[c] = clfs[c].fit_predict(pdf[[c]].fillna(median))

            return _pdf

        # Sets a grouping key for inference
        num_parallelism = self._spark.sparkContext.defaultParallelism
        grouping_key = self._create_temp_name("grouping_key")
        input_df = input_df.withColumn(grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))
        predicted_df = input_df.groupBy(grouping_key).apply(predict)

        def _extract_err_cells(col: str) -> Any:
            return predicted_df.where(f'{col} < 0').selectExpr(f'{self.row_id}', f'"{col}" attribute')

        sdfs = list(map(lambda c: _extract_err_cells(c), columns))
        return functools.reduce(lambda x, y: x.union(y), sdfs)


class LOFOutlierErrorDetector(ScikitLearnBasedErrorDetector):

    def __init__(self, parallel_mode_threshold: int = 10000) -> None:
        ScikitLearnBasedErrorDetector.__init__(self, parallel_mode_threshold)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _outlier_detector_impl(self) -> Any:
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(novelty=False)
