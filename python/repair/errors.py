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

import functools
import json
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession, functions  # type: ignore
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from repair.utils import get_option_value, get_random_string, setup_logger, \
    spark_job_group, to_list_str


_logger = setup_logger()


class ErrorDetector(metaclass=ABCMeta):

    def __init__(self, targets: List[str] = []) -> None:
        self.row_id: Optional[str] = None
        self.qualified_input_name: Optional[str] = None
        self.continous_cols: List[str] = []
        self.targets: List[str] = targets

        # For Spark/JVM interactions
        self._spark = SparkSession.builder.getOrCreate()
        self._detector_api = self._spark.sparkContext._active_spark_context._jvm.ErrorDetectorApi  # type: ignore

    def setUp(self, row_id: str, qualified_input_name: str,
              continous_cols: List[str], targets: List[str]) -> "ErrorDetector":
        self.row_id = row_id
        self.qualified_input_name = qualified_input_name
        self.continous_cols = continous_cols
        if self.targets:
            self._targets = list(set(self.targets) & set(targets))
        else:
            self._targets = targets

        return self

    @abstractmethod
    def _detect_impl(self) -> DataFrame:
        pass

    def _to_continous_col_list(self) -> str:
        return ','.join(self.continous_cols) if self.continous_cols else ''

    def _to_target_list(self) -> str:
        assert hasattr(self, '_targets'), '`setUp` should be called before `_to_target_list`'
        return ','.join(self._targets) if self._targets else ''

    def _empty_dataframe(self) -> DataFrame:
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
            return self._empty_dataframe()

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

    def __init__(self, constraint_path: str, targets: List[str] = []) -> None:
        ErrorDetector.__init__(self, targets)
        self.constraint_path = constraint_path

    def __str__(self) -> str:
        param_targets = f',targets={",".join(self.targets)}' if self.targets else ''
        params = f'path={self.constraint_path}{param_targets}'
        return f'{self.__class__.__name__}({params})'

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

    def __init__(self, parallel_mode_threshold: int = 10000, num_parallelism: Optional[int] = None) -> None:
        ErrorDetector.__init__(self)
        self.parallel_mode_threshold = parallel_mode_threshold
        self.num_parallelism = num_parallelism
        if num_parallelism is not None and int(num_parallelism) <= 0:
            raise ValueError(f'`num_parallelism` must be positive, got {num_parallelism}')

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    @property
    def _num_parallelism(self) -> int:
        return int(self.num_parallelism) if self.num_parallelism \
            else self._spark.sparkContext.defaultParallelism

    # NOTE: An overrode method must return an instance having a scikit-learn-like `fit_predict(X)` method;
    # the method fits a model to the training set X and return labels.
    # The predicted labels are 1(inlier) or -1(outlier) of X.
    @abstractmethod
    def _outlier_detector_impl(self) -> Any:
        pass

    def _detect_impl(self) -> DataFrame:
        columns = list(filter(lambda c: c in self._targets, self.continous_cols)) if self._targets \
            else self.continous_cols
        if not columns:
            return self._empty_dataframe()

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
                else self._empty_dataframe()

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
        grouping_key = get_random_string("grouping_key")
        grouping_key_expr = functions.rand() * functions.lit(self._num_parallelism)
        input_df = input_df.withColumn(grouping_key, grouping_key_expr.cast("int"))
        predicted_df = input_df.groupBy(grouping_key).apply(predict)

        def _extract_err_cells(col: str) -> Any:
            return predicted_df.where(f'{col} < 0').selectExpr(f'{self.row_id}', f'"{col}" attribute')

        sdfs = list(map(lambda c: _extract_err_cells(c), columns))
        return functools.reduce(lambda x, y: x.union(y), sdfs)


class ScikitLearnBackedErrorDetector(ScikitLearnBasedErrorDetector):

    def __init__(self, error_detector_cls: Callable[[], Any], parallel_mode_threshold: int = 10000,
                 num_parallelism: Optional[int] = None) -> None:
        ScikitLearnBasedErrorDetector.__init__(self, parallel_mode_threshold, num_parallelism)

        if not hasattr(error_detector_cls, "__call__"):
            raise ValueError('`error_detector_cls` should be callable')
        if not hasattr(error_detector_cls(), "fit_predict"):
            raise ValueError('An instance that `error_detector_cls` returns should have a `fit_predict` method')

        self.error_detector_cls = error_detector_cls

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _outlier_detector_impl(self) -> Any:
        return self.error_detector_cls()


class LOFOutlierErrorDetector(ScikitLearnBasedErrorDetector):

    def __init__(self, parallel_mode_threshold: int = 10000, num_parallelism: Optional[int] = None) -> None:
        ScikitLearnBasedErrorDetector.__init__(self, parallel_mode_threshold, num_parallelism)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _outlier_detector_impl(self) -> Any:
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(novelty=False)


class ErrorModel():

    # List of internal configurations
    from collections import namedtuple
    _option = namedtuple('_option', 'key default_value type_class validator err_msg')

    _opt_freq_attr_stat_threshold = \
        _option('error.freq_attr_stat_threshold', 0.0, float,
                lambda v: 0.0 <= v and v <= 1.0, '`{}` should be in [0.0, 1.0]')
    _opt_max_attrs_to_compute_domains = \
        _option('error.max_attrs_to_compute_domains', 4, int,
                lambda v: v >= 2, '`{}` should be greater than 1')
    _opt_domain_threshold_alpha = \
        _option('error.domain_threshold_alpha', 0.0, float,
                lambda v: 0.0 <= v and v < 1.0, '`{}` should be in [0.0, 1.0)')
    _opt_domain_threshold_beta = \
        _option('error.domain_threshold_beta', 0.70, float,
                lambda v: 0.0 <= v and v < 1.0, '`{}` should be in [0.0, 1.0)')

    option_keys = set([
        _opt_freq_attr_stat_threshold.key,
        _opt_max_attrs_to_compute_domains.key,
        _opt_domain_threshold_alpha.key,
        _opt_domain_threshold_beta.key])

    def __init__(self, row_id: str, targets: List[str], discrete_thres: int,
                 error_detectors: List[ErrorDetector],
                 error_cells: Optional[str],
                 opts: Dict[str, str]) -> None:
        self.row_id: str = str(row_id)
        self.targets: List[str] = targets
        self.discrete_thres: int = discrete_thres
        self.error_detectors: List[ErrorDetector] = error_detectors
        self.error_cells: Optional[str] = error_cells

        # Options for internal behaviours
        self.opts: Dict[str, str] = opts

        # Temporary views to keep intermediate results; these views are automatically
        # created when repairing data, and then dropped finally.
        self._intermediate_views_on_runtime: List[str] = []

        # JVM interfaces for Data Repair/Graph APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._jvm = self._spark.sparkContext._active_spark_context._jvm  # type: ignore
        self._repair_api = self._jvm.RepairApi

    def _get_option_value(self, *args) -> Any:  # type: ignore
        return get_option_value(self.opts, *args)

    def _delete_view_on_exit(self, view_name: str) -> None:
        self._intermediate_views_on_runtime.append(view_name)

    def _create_temp_view(self, df: DataFrame, prefix: str) -> str:
        assert isinstance(df, DataFrame)
        view_name = get_random_string(prefix)
        df.createOrReplaceTempView(view_name)
        self._delete_view_on_exit(view_name)
        return view_name

    def _release_resources(self) -> None:
        while self._intermediate_views_on_runtime:
            v = self._intermediate_views_on_runtime.pop()
            _logger.debug(f"Dropping an auto-generated view: {v}")
            self._spark.sql(f"DROP VIEW IF EXISTS {v}")

    def _get_default_error_detectors(self, input_table: str) -> List[ErrorDetector]:
        error_detectors: List[ErrorDetector] = [NullErrorDetector()]
        targets = self.targets if self.targets else \
            [c for c in self._spark.table(input_table).columns if c != self.row_id]
        for c in targets:
            error_detectors.append(DomainValues(attr=c, autofill=True, min_count_thres=4))

        return error_detectors

    def _target_attrs(self, input_columns: List[str]) -> List[str]:
        target_attrs = list(filter(lambda c: c != self.row_id, input_columns))
        if self.targets:
            target_attrs = list(set(self.targets) & set(target_attrs))  # type: ignore
        return target_attrs

    # TODO: Needs to implement an error detector based on edit distances
    def _detect_error_cells(self, input_table: str, continous_columns: List[str]) -> DataFrame:
        error_detectors = self.error_detectors
        if not error_detectors:
            error_detectors = self._get_default_error_detectors(input_table)

        _logger.info(f'[Error Detection Phase] Used error detectors: {to_list_str(error_detectors)}')

        # Computes target attributes for error detection
        target_attrs = self._target_attrs(self._spark.table(input_table).columns)

        # Initializes the given error detectors with the input params
        for d in error_detectors:
            d.setUp(self.row_id, input_table, continous_columns, target_attrs)  # type: ignore

        error_cells_dfs = [d.detect() for d in error_detectors]
        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        return err_cells_df.distinct().cache()

    def _with_current_values(self, input_table: str, noisy_cells_df: DataFrame, targetAttrs: List[str]) -> DataFrame:
        noisy_cells = self._create_temp_view(noisy_cells_df, "noisy_cells_v1")
        jdf = self._repair_api.withCurrentValues(input_table, noisy_cells, self.row_id, ','.join(targetAttrs))
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def _filter_columns_from(self, df: DataFrame, targets: List[str], negate: bool = False) -> DataFrame:
        return df.where("attribute {} ({})".format("NOT IN" if negate else "IN", to_list_str(targets, quote=True)))

    @spark_job_group(name="error detection")
    def _detect_errors(self, input_table: str, continous_columns: List[str]) -> Tuple[DataFrame, List[str]]:
        # If `self.error_cells` provided, just uses it
        if self.error_cells is not None:
            # TODO: Even in this case, we need to use a NULL detector because
            # `_build_stat_model` will fail if `y` has NULL.
            noisy_cells_df = self._spark.table(self.error_cells)
            _logger.info(f'[Error Detection Phase] Error cells provided by `{self.error_cells}`')

            if len(self.targets) == 0:
                # Filters out non-existent columns in `input_table`
                noisy_cells_df = self._filter_columns_from(
                    noisy_cells_df, self._spark.table(input_table).columns)
            else:
                # Filters target attributes if `self.targets` defined
                noisy_cells_df = self._filter_columns_from(noisy_cells_df, self.targets)
        else:
            # Applies error detectors to get noisy cells
            noisy_cells_df = self._detect_error_cells(input_table, continous_columns)

        noisy_columns: List[str] = []
        num_noisy_cells = noisy_cells_df.count()
        if num_noisy_cells > 0:
            noisy_columns = noisy_cells_df \
                .selectExpr("collect_set(attribute) columns") \
                .collect()[0] \
                .columns
            noisy_cells_df = self._with_current_values(
                input_table, noisy_cells_df, noisy_columns)

        return noisy_cells_df, noisy_columns

    @spark_job_group(name="cell domain analysis")
    def _analyze_error_cell_domain(
            self, discretized_table: str, noisy_cells_df: DataFrame,
            continous_columns: List[str], target_columns: List[str],
            domain_stats: Dict[str, int],
            freq_attr_stats: str,
            pairwise_attr_corr_stats: Dict[str, int]) -> str:
        _logger.info("[Error Detection Phase] Analyzing cell domains to fix error cells...")

        noisy_cells = self._create_temp_view(noisy_cells_df, "noisy_cells_v3")
        jdf = self._repair_api.computeDomainInErrorCells(
            discretized_table, noisy_cells, self.row_id,
            ",".join(continous_columns),
            ",".join(target_columns),
            freq_attr_stats,
            json.dumps(pairwise_attr_corr_stats),
            json.dumps(domain_stats),
            self._get_option_value(*self._opt_max_attrs_to_compute_domains),
            self._get_option_value(*self._opt_domain_threshold_alpha),
            self._get_option_value(*self._opt_domain_threshold_beta))

        cell_domain_df = DataFrame(jdf, self._spark._wrapped)  # type: ignore
        cell_domain = self._create_temp_view(cell_domain_df.cache(), "cell_domain")
        return cell_domain

    def _compute_attr_stats(self, discretized_table: str, target_columns: List[str],
                            domain_stats: Dict[str, int]) -> Tuple[str, Dict[str, Any]]:
        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        ret_as_json = json.loads(self._repair_api.computeAttrStats(
            discretized_table,
            self.row_id,
            ','.join(target_columns),
            json.dumps(domain_stats),
            self._get_option_value(*self._opt_freq_attr_stat_threshold)))

        freq_attr_stats = ret_as_json['freq_attr_stats']
        pairwise_attr_corr_stats = ret_as_json['pairwise_attr_corr_stats']
        self._delete_view_on_exit(freq_attr_stats)

        return freq_attr_stats, pairwise_attr_corr_stats

    def _extract_error_cells_from(self, noisy_cells_df: DataFrame, input_table: str,
                                  discretized_table: str, continous_columns: List[str], target_columns: List[str],
                                  pairwise_attr_corr_stats: Dict[str, Tuple[str, float]],
                                  freq_attr_stats: str,
                                  domain_stats: Dict[str, int]) -> DataFrame:
        # Defines true error cells based on the result of domain analysis
        cell_domain = self._analyze_error_cell_domain(
            discretized_table, noisy_cells_df, continous_columns, target_columns,
            domain_stats, freq_attr_stats, pairwise_attr_corr_stats)

        # Fixes cells if a predicted value is the same with an initial one
        fix_cells_expr = "if(current_value = domain[0].n, current_value, NULL) repaired"
        weak_labeled_cells_df = self._spark.table(cell_domain) \
            .selectExpr(f"`{self.row_id}`", "attribute", fix_cells_expr) \
            .where("repaired IS NOT NULL")

        # Removes weak labeled cells from the noisy cells
        error_cells_df = noisy_cells_df.join(weak_labeled_cells_df, [self.row_id, "attribute"], "left_anti")
        assert noisy_cells_df.count() == error_cells_df.count() + weak_labeled_cells_df.count()

        _logger.info('[Error Detection Phase] {} noisy cells fixed and '
                     '{} error cells remaining...'.format(weak_labeled_cells_df.count(), error_cells_df.count()))

        return error_cells_df

    # Checks if attributes are discrete or not, and discretizes continous ones
    def _discretize_attrs(self, input_table: str) -> Tuple[str, Dict[str, int]]:
        # Filters out attributes having large domains and makes continous values
        # discrete if necessary.
        ret_as_json = json.loads(self._repair_api.convertToDiscretizedTable(
            input_table, self.row_id, self.discrete_thres))

        discretized_table = ret_as_json["discretized_table"]
        self._delete_view_on_exit(discretized_table)

        domain_stats = {k: int(v) for k, v in ret_as_json["domain_stats"].items()}
        return discretized_table, domain_stats

    def detect(self, input_table: str, continous_columns: List[str]) \
            -> Tuple[DataFrame, List[str], Dict[str, Any], Dict[str, int]]:
        try:
            # If no error found, we don't need to do nothing
            noisy_cells_df, noisy_columns = self._detect_errors(input_table, continous_columns)
            if noisy_cells_df.count() == 0:  # type: ignore
                return noisy_cells_df, [], {}, {}

            discretized_table, domain_stats = self._discretize_attrs(input_table)
            discretized_columns = self._spark.table(discretized_table).columns
            if len(discretized_columns) == 0:
                return noisy_cells_df, [], {}, {}

            # Target repairable(discretizable) columns
            target_columns = list(filter(lambda c: c in discretized_columns, noisy_columns))

            # Cannot compute pair-wise stats when `len(discretized_columns) <= 1`
            if len(target_columns) == 0 or len(discretized_columns) <= 1:
                return noisy_cells_df, target_columns, {}, domain_stats

            # Computes attribute stats for the discretized table
            freq_attr_stats, pairwise_attr_corr_stats = self._compute_attr_stats(
                discretized_table, target_columns, domain_stats)

            error_cells_df = noisy_cells_df
            if not self.error_cells:
                error_cells_df = self._extract_error_cells_from(
                    noisy_cells_df, input_table, discretized_table,
                    continous_columns, target_columns,
                    pairwise_attr_corr_stats,
                    freq_attr_stats,
                    domain_stats)

            return error_cells_df, target_columns, pairwise_attr_corr_stats, domain_stats
        except:
            raise
        finally:
            self._release_resources()
