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

import os
import copy
import datetime
import functools
import heapq
import json
import logging
import pickle
import numpy as np   # type: ignore[import]
import pandas as pd  # type: ignore[import]
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame, Row, SparkSession, functions  # type: ignore[import]
from pyspark.sql.functions import col, expr  # type: ignore[import]

from repair.detectors import ErrorDetector, NullErrorDetector
from repair.distances import Distance, Levenshtein
from repair.utils import argtype_check, elapsed_time


class RepairModel():
    """
    Interface to detect error cells in given input data and build a statistical
    model to repair them.

    .. versionchanged:: 0.1.0
    """

    def __init__(self) -> None:
        super().__init__()

        # Basic parameters
        self.db_name: str = ""
        self.input: Optional[Union[str, DataFrame]] = None
        self.row_id: Optional[str] = None
        self.targets: List[str] = []

        # For storing built models and their params into a persistent storage
        # to analyze the training process.
        self.checkpoint_path: Optional[str] = None

        # Parameters for error detection
        self.error_cells: Optional[Union[str, DataFrame]] = None
        # To find error cells, the NULL detector is used by default
        self.error_detectors: List[ErrorDetector] = [NullErrorDetector()]
        self.discrete_thres: int = 80
        self.min_corr_thres: float = 0.70
        self.domain_threshold_alpha: float = 0.0
        self.domain_threshold_beta: float = 0.70
        self.max_attrs_to_compute_domains: int = 4
        self.attr_stat_sample_ratio: float = 1.0
        self.attr_stat_threshold: float = 0.0

        # Parameters for repair model training
        self.training_data_sample_ratio: float = 1.0
        self.min_training_row_ratio: float = 0.10
        self.max_training_column_num: Optional[int] = None
        self.small_domain_threshold: int = 12
        self.inference_order: str = "entropy"

        # Parameters for repairing
        self.maximal_likelihood_repair_enabled: bool = False
        self.repair_delta: Optional[int] = None

        # Defines a class to compute cost of updates.
        #
        # TODO: Needs a sophisticated way to compute distances between a current value and a repair
        # candidate. For example, the HoloDetect paper [1] proposes a noisy channel model for the
        # data augmentation methodology of training data. This model consists of transformation rule
        # and and data augmentation policies (i.e., distribution over those data transformation).
        # This model can be re-used to compute this cost. For more details, see the section 5,
        # 'DATA AUGMENTATION LEARNING', in the paper.
        self.distance: Distance = Levenshtein()

        # Options for internal behaviours
        self.opts: Dict[str, str] = {}

        # Temporary views to keep intermediate results; these views are automatically
        # created when repairing data, and then dropped finally.
        #
        # TODO: Move this variable into a runtime `env`
        self._intermediate_views_on_runtime: List[str] = []

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._jvm = self._spark.sparkContext._active_spark_context._jvm
        self._repair_api = self._jvm.RepairApi

    @argtype_check  # type: ignore
    def setDbName(self, db_name: str) -> "RepairModel":
        """Specifies the database name for an input table.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        db_name : str
            database name (default: '').
        """
        if type(self.input) is DataFrame:
            raise ValueError("Can not specify a database name when input is `DataFrame`")
        self.db_name = db_name
        return self

    @argtype_check  # type: ignore
    def setTableName(self, table_name: str) -> "RepairModel":
        """Specifies the table or view name to repair data.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        table_name : str
            table or view name.
        """
        self.input = table_name
        return self

    @argtype_check  # type: ignore
    def setInput(self, input: Union[str, DataFrame]) -> "RepairModel":
        """Specifies the table/view name or :class:`DataFrame` to repair data.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        input: str, :class:`DataFrame`
            table/view name or :class:`DataFrame`.
        """
        if type(input) is DataFrame:
            self.db_name = ""
        self.input = input
        return self

    @argtype_check  # type: ignore
    def setRowId(self, row_id: str) -> "RepairModel":
        """Specifies the table name or :class:`DataFrame` to repair data.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        input: str
            the column where all values are different.
        """
        self.row_id = row_id
        return self

    @argtype_check  # type: ignore
    def setTargets(self, attrs: List[str]) -> "RepairModel":
        """Specifies target attributes to repair.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        attrs: list
            list of target attributes.
        """
        if len(attrs) == 0:
            raise ValueError("`attrs` has at least one attribute")
        self.targets = attrs
        return self

    @argtype_check  # type: ignore
    def setCheckpointPath(self, path: str) -> "RepairModel":
        """
        Specifies a directory path to store built models and their params into a persistent
        storage to analyze the training process.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        path: str
            directory path for checkpointing (default: ``None``).
        """
        if os.path.exists(path):
            raise ValueError(f"Path '{path}' already exists")
        self.checkpoint_path = path
        return self

    @argtype_check  # type: ignore
    def setErrorCells(self, error_cells: Union[str, DataFrame]) -> "RepairModel":
        """Specifies the table/view name or :class:`DataFrame` defining where error cells are.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        input: str, :class:`DataFrame`
            table/view name or :class:`DataFrame`.

        Examples
        --------
        >>> spark.table("error_cells").show()
        +---+---------+
        |tid|attribute|
        +---+---------+
        |  3|      Sex|
        | 12|      Age|
        | 16|   Income|
        +---+---------+

        >>> df = scavenger.repair.setInput("adult").setRowId("tid")
        ...     .setErrorCells("error_cells").run()
        >>> df.show()
        +---+---------+-------------+-----------+
        |tid|attribute|current_value|   repaired|
        +---+---------+-------------+-----------+
        |  3|      Sex|         null|     Female|
        | 12|      Age|         null|      18-21|
        | 16|   Income|         null|MoreThan50K|
        +---+---------+-------------+-----------+
        """
        self.error_cells = error_cells
        return self

    @argtype_check  # type: ignore
    def setErrorDetector(self, detector: ErrorDetector) -> "RepairModel":
        """
        Specifies the :class:`ErrorDetector` derived class to implement
        a logic to detect error cells.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        detector: derived class of :class:`ErrorDetector`
            specifies how to detect error cells. Available classes are as follows:

            * :class:`RegExErrorDetector`: detects error cells based on a regular expresson.
            * :class:`OutlierErrorDetector`: detects error cells based on the Gaussian distribution.
            * :class:`ConstraintErrorDetector`: detects error cells based on integrity rules
              defined by denial constraints.
        """
        if not isinstance(detector, ErrorDetector):
            raise TypeError("`detector` should be provided as repair.detectors.ErrorDetector, "
                            "got {type(detector)}")
        self.error_detectors.append(detector)
        return self

    @argtype_check  # type: ignore
    def setDiscreteThreshold(self, thres: int) -> "RepairModel":
        """Specifies max domain size of discrete values.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        thres: int
            max domain size of discrete values. The values must be bigger than 1 and
            the default value is 80.
        """
        if int(thres) < 2:
            raise ValueError("threshold must be bigger than 1")
        self.discrete_thres = thres
        return self

    @argtype_check  # type: ignore
    def setMinCorrThreshold(self, thres: float) -> "RepairModel":
        """Specifies a threshold to decide which columns are used to compute domains.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        thres: float
           threshold value. The value must be in [0.0, 1.0] and
           the default value is 0.7.0.
        """
        self.min_corr_thres = thres
        return self

    @argtype_check  # type: ignore
    def setDomainThresholds(self, alpha: float, beta: float) -> "RepairModel":
        """Specifies a thresholds to reduce domain size.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        thres: float
           threshold values. The values must be in [0.0, 1.0] and
           the default values of alpha and beta are 0.0 and 0.70, respectively.
        """
        self.domain_threshold_alpha = alpha
        self.domain_threshold_beta = beta
        return self

    @argtype_check  # type: ignore
    def setAttrMaxNumToComputeDomains(self, max: int) -> "RepairModel":
        """
        Specifies the max number of attributes to compute posterior probabiity
        based on the Naive Bayes assumption.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        thres: int
            the max number of attributes (default: 4).
        """
        self.max_attrs_to_compute_domains = max
        return self

    @argtype_check  # type: ignore
    def setAttrStatSampleRatio(self, ratio: float) -> "RepairModel":
        """Specifies a sample ratio for table used to compute co-occurrence frequency.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        ratio: float
            sampling ratio (default: 1.0).
        """
        self.attr_stat_sample_ratio = ratio
        return self

    @argtype_check  # type: ignore
    def setAttrStatThreshold(self, ratio: float) -> "RepairModel":
        """Specifies a threshold for filtering out low frequency.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        ratio: float
            threshold value (default: 0.0).
        """
        self.attr_stat_threshold = ratio
        return self

    @argtype_check  # type: ignore
    def setTrainingDataSampleRatio(self, ratio: float) -> "RepairModel":
        """Specifies a sample ratio for table used to build statistical models.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        ratio: float
            sampling ratio (default: 1.0).
        """
        self.training_data_sample_ratio = ratio
        return self

    @argtype_check  # type: ignore
    def setMaxTrainingColumnNum(self, n: int) -> "RepairModel":
        """Specifies the max number of columns used to build models.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        n: int
            the max number of columns (default: None).
        """
        self.max_training_column_num = n
        return self

    @argtype_check  # type: ignore
    def setSmallDomainThreshold(self, thres: int) -> "RepairModel":
        """Specifies max domain size for low-cardinality catogory encoding.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        thres: int
            threshold value (default: 12).
        """
        self.small_domain_threshold = thres
        return self

    @argtype_check  # type: ignore
    def setInferenceOrder(self, inference_order: str) -> "RepairModel":
        """Specifies how to order target columns when building models.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        inference_order: str
            built-in logic name (default: 'entropy').
        """
        self.inference_order = inference_order
        return self

    @argtype_check  # type: ignore
    def setMaximalLikelihoodRepairEnabled(self, enabled: bool) -> "RepairModel":
        """Specifies whether to enable maximal likelihood repair.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        enabled: bool
            If set to ``True``, uses maximal likelihood repair (default: ``False``).
        """
        self.maximal_likelihood_repair_enabled = enabled
        return self

    @argtype_check  # type: ignore
    def setRepairDelta(self, delta: int) -> "RepairModel":
        """Specifies the max number of applied repairs.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        delta: int
            delta value (default: None). The value must be positive.
        """
        if delta <= 0:
            raise ValueError("Repair delta must be positive")
        self.repair_delta = int(delta)
        return self

    @argtype_check  # type: ignore
    def option(self, key: str, value: str) -> "RepairModel":
        """Adds an input option for internal functionalities (e.g., model learning).

        .. versionchanged:: 0.1.0
        """
        self.opts[key] = value
        return self

    @property
    def _input_table(self) -> str:
        return self._create_temp_view(self.input, "input") if type(self.input) is DataFrame \
            else str(self.input)

    @property
    def _error_cells(self) -> str:
        df = self.error_cells if type(self.error_cells) is DataFrame \
            else self._spark.table(str(self.error_cells))
        if not all(c in df.columns for c in (self.row_id, "attribute")):  # type: ignore
            raise ValueError(f"Error cells must have `{self.row_id}` and "
                             "`attribute` in columns")
        return self._create_temp_view(df, "error_cells")

    def _get_option(self, key: str, default_value: str) -> str:
        return self.opts[str(key)] if str(key) in self.opts else default_value

    def _clear_job_group(self) -> None:
        # TODO: Uses `SparkContext.clearJobGroup()` instead
        self._spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        self._spark.sparkContext.setLocalProperty("spark.job.description", None)
        self._spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)

    def _spark_job_group(name: str):  # type: ignore
        def decorator(f):  # type: ignore
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):  # type: ignore
                self._spark.sparkContext.setJobGroup(name, name)
                start_time = time.time()
                ret = f(self, *args, **kwargs)
                logging.info(f"Elapsed time (name: {name}) is {time.time() - start_time}(s)")
                self._clear_job_group()

                return ret
            return wrapper
        return decorator

    def _register_and_get_df(self, view_name: str) -> DataFrame:
        self._intermediate_views_on_runtime.append(view_name)
        return self._spark.table(view_name)

    def _create_temp_view(self, df: DataFrame, prefix: str = "temp") -> str:
        temp_name = f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        df.createOrReplaceTempView(temp_name)
        self._intermediate_views_on_runtime.append(temp_name)
        return temp_name

    def _repair_attrs(self, repair_updates: str, base_table: str) -> DataFrame:
        jdf = self._jvm.RepairMiscApi.repairAttrsFrom(
            repair_updates, "", base_table, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)

    def _flatten(self, input_table: str) -> DataFrame:
        jdf = self._jvm.RepairMiscApi.flattenTable("", input_table, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)

    def _release_resources(self) -> None:
        while self._intermediate_views_on_runtime:
            v = self._intermediate_views_on_runtime.pop()
            logging.debug(f"Dropping an auto-generated view: {v}")
            self._spark.sql(f"DROP VIEW IF EXISTS {v}")

    def _check_input_table(self, env: Dict[str, str]) -> Tuple[str, List[str]]:
        ret_as_json = self._repair_api.checkInputTable(
            self.db_name, self._input_table, self.row_id)
        env.update(json.loads(ret_as_json))
        continous_attrs = env["continous_attrs"].split(",")
        return self._spark.table(env["input_table"]), \
            continous_attrs if continous_attrs != [""] else []

    def _detect_error_cells(self, input_table: str) -> str:
        # Initializes the given error detectors with the input params
        for d in self.error_detectors:
            d.setUp(self.row_id, input_table)  # type: ignore

        error_cells_dfs = [d.detect() for d in self.error_detectors]

        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        err_cells = self._create_temp_view(err_cells_df.distinct().cache(), "error_cells")
        return err_cells

    @_spark_job_group(name="error detection")
    def _detect_errors(self, env: Dict[str, str]) -> DataFrame:
        # If `self.error_cells` provided, just uses it
        if self.error_cells is not None:
            # TODO: Even in this case, we need to use a NULL detector because
            # `_build_model` will fial if `y` has NULL.
            gray_cells_view = self._error_cells
            logging.info(f'[Error Detection Phase] Error cells provided by `{gray_cells_view}`')

            # We assume that the given error cells are true, so we skip computing error domains
            # with probability because the computational cost is much high.
            self.domain_threshold_beta = 1.0
        else:
            # Applys error detectors to get gray cells
            gray_cells_view = self._detect_error_cells(env["input_table"])
            logging.info(f'[Error Detection Phase] Detecting errors '
                         f'in a table `{env["input_table"]}` '
                         f'({env["num_attrs"]} cols x {env["num_input_rows"]} rows)...')

        # Filters target attributes if `self.targets` defined
        if len(self.targets) > 0:
            in_list = ",".join(map(lambda x: f"'{x}'", self.targets))
            df = self._spark.sql("SELECT * FROM {} WHERE attribute IN ({})".format(
                gray_cells_view, in_list))
            env["gray_cells"] = self._create_temp_view(df, "gray_cells")
        else:
            env["gray_cells"] = gray_cells_view

        return self._spark.table(env["gray_cells"])

    def _prepare_repair_base(self, env: Dict[str, str], gray_cells_df: DataFrame) -> DataFrame:
        # Sets NULL at the detected gray cells
        logging.debug("{}/{} suspicious cells found, then converts them into NULL cells...".format(
            gray_cells_df.count(), int(env["num_input_rows"]) * int(env["num_attrs"])))
        env.update(json.loads(self._repair_api.convertErrorCellsToNull(
            env["input_table"], env["gray_cells"],
            self.row_id)))

        return self._register_and_get_df(env["repair_base"])

    def _preprocess(self, env: Dict[str, str], continous_attrs: List[str]) -> DataFrame:
        # Filters out attributes having large domains and makes continous values
        # discrete if necessary.
        env.update(json.loads(self._repair_api.convertToDiscreteFeatures(
            env["input_table"], self.row_id, self.discrete_thres)))

        discrete_ft_df = self._register_and_get_df(env["discrete_features"])
        logging.debug("Valid {} attributes ({}) found in the {} input attributes ({}) and "
                      "{} continous attributes ({}) included in them".format(
                          len(discrete_ft_df.columns),
                          ",".join(discrete_ft_df.columns),
                          len(self._spark.table(env["input_table"]).columns),
                          ",".join(self._spark.table(env["input_table"]).columns),
                          len(continous_attrs),
                          ",".join(continous_attrs)))

        return discrete_ft_df

    def _compute_attr_stats(self, env: Dict[str, str]) -> str:
        ret = json.loads(self._repair_api.computeAttrStats(
            env["discrete_features"], env["gray_cells"], self.row_id,
            self.attr_stat_sample_ratio,
            self.attr_stat_threshold))
        self._intermediate_views_on_runtime.append(ret["attr_stats"])
        return ret["attr_stats"]

    @_spark_job_group(name="cell domain analysis")
    def _analyze_error_cell_domain(self, env: Dict[str, str], gray_cells_df: DataFrame,
                                   continous_attrs: List[str]) -> str:
        # Checks if attributes are discrete or not, and discretizes continous ones
        discrete_ft_df = self._preprocess(env, continous_attrs)

        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        logging.debug("Collecting and sampling attribute stats (ratio={} threshold={}) "
                      "before computing error domains...".format(
                          self.attr_stat_sample_ratio,
                          self.attr_stat_threshold))

        attr_stats = self._compute_attr_stats(env)

        logging.info("[Error Detection Phase] Analyzing cell domains to fix error cells...")
        env.update(json.loads(self._repair_api.computeDomainInErrorCells(
            env["discrete_features"], attr_stats, env["gray_cells"], self.row_id,
            env["continous_attrs"],
            self.max_attrs_to_compute_domains,
            self.min_corr_thres,
            self.domain_threshold_alpha,
            self.domain_threshold_beta)))

        return self._register_and_get_df(env["cell_domain"])

    def _extract_error_cells(self, env: Dict[str, str], cell_domain_df: DataFrame,
                             repair_base_df: DataFrame) -> DataFrame:
        # Fixes cells if an inferred value is the same with an initial one
        fix_cells_expr = "if(current_value = domain[0].n, current_value, NULL) value"
        weak_df = cell_domain_df.selectExpr(
            self.row_id, "attribute", "current_value", fix_cells_expr).cache()
        error_cells_df = weak_df.where("value IS NULL").drop("value").cache()
        weak_df = weak_df.where("value IS NOT NULL") \
            .selectExpr(self.row_id, "attribute", "value repaired")
        env["weak"] = self._create_temp_view(weak_df.cache(), "weak")
        env["partial_repaired"] = self._create_temp_view(
            self._repair_attrs(env["weak"], env["repair_base"]),
            "partial_repaired")

        logging.info('[Error Detection Phase] {} suspicious cells fixed and '
                     '{} error cells remaining...'.format(
                         self._spark.table(env["weak"]).count(),
                         error_cells_df.count()))

        return error_cells_df

    def _split_clean_and_dirty_rows(
            self, env: Dict, error_cells_df: DataFrame) -> Tuple[DataFrame, DataFrame, List[Row]]:
        error_rows_df = error_cells_df.selectExpr(self.row_id).distinct().cache()
        fixed_df = self._spark.table(env["partial_repaired"]) \
            .join(error_rows_df, self.row_id, "left_anti").cache()
        dirty_df = self._spark.table(env["partial_repaired"]) \
            .join(error_rows_df, self.row_id, "left_semi").cache()
        error_attrs = error_cells_df.groupBy("attribute") \
            .agg(functions.count("attribute").alias("cnt")).collect()
        assert len(error_attrs) > 0
        return fixed_df, dirty_df, error_attrs

    def _convert_to_histogram(self, df: DataFrame) -> DataFrame:
        input_table = self._create_temp_view(df)
        ret_as_json = self._repair_api.convertToHistogram(input_table, self.discrete_thres)
        return self._spark.table(json.loads(ret_as_json)["histogram"])

    def _show_histogram(self, df: DataFrame) -> None:
        import matplotlib.pyplot as plt  # type: ignore[import]
        fig = plt.figure()
        num_targets = df.count()
        for index, row in enumerate(df.collect()):
            pdf = df.where(f'attribute = "{row.attribute}"') \
                .selectExpr("inline(histogram)").toPandas()
            f = fig.add_subplot(num_targets, 1, index + 1)
            f.bar(pdf["value"], pdf["cnt"])
            f.set_xlabel(row.attribute)
            f.set_ylabel("cnt")

        fig.tight_layout()
        fig.show()

    def _select_training_rows(self, fixed_df: DataFrame) -> DataFrame:
        # Prepares training data to repair the remaining error cells
        # TODO: Needs more smart sampling, e.g., down-sampling
        train_df = fixed_df.sample(self.training_data_sample_ratio).drop(self.row_id).cache()
        logging.info("[Repair Model Training Phase] Sampling {} training data (ratio={}) "
                     "from {} clean rows...".format(
                         train_df.count(),
                         self.training_data_sample_ratio,
                         fixed_df.count()))
        return train_df

    def _error_num_based_order(self, error_attrs: List[Row]) -> List[str]:
        # Sorts target columns by the number of errors
        error_num_map = {}
        for row in error_attrs:
            error_num_map[row.attribute] = row.cnt

        target_columns = list(map(lambda row: row.attribute,
                              sorted(error_attrs, key=lambda row: row.cnt, reverse=False)))
        for y in target_columns:
            logging.debug(f"{y}: #errors={error_num_map[y]}")

        return target_columns

    @_spark_job_group(name="training data stat analysis")
    def _domain_size_based_order(self, env: Dict[str, str], train_df: DataFrame,
                                 error_attrs: List[Row]) -> List[str]:
        # Computes domain sizes for training data
        logging.info("[Repair Model Training Phase] Collecting training data stats before "
                     "building ML models...")
        env["train"] = self._create_temp_view(train_df, "train")
        env.update(json.loads(self._repair_api.computeDomainSizes(env["train"])))

        # Sorts target columns by domain size
        target_columns = list(map(lambda row: row.attribute,
                              sorted(error_attrs,
                                     key=lambda row: int(env["distinct_stats"][row.attribute]),
                                     reverse=False)))
        for y in target_columns:
            logging.debug(f'{y}: |domain|={env["distinct_stats"][y]}')

        return target_columns

    def _entropy_based_order(self, env: Dict[str, str], train_df: DataFrame,
                             error_attrs: List[Row]) -> List[str]:
        # Sorts target columns by correlations
        target_columns: List[str] = []
        error_attrs = list(map(lambda row: row.attribute, error_attrs))

        for index in range(len(error_attrs)):
            features = [c for c in train_df.columns if c not in error_attrs]
            targets: List[Tuple[float, str]] = []
            for c in error_attrs:
                total_corr = 0.0
                for f, corr in map(lambda x: tuple(x), env["pairwise_attr_stats"][c]):
                    if f in features:
                        total_corr += float(corr)

                heapq.heappush(targets, (-total_corr, c))

            t = heapq.heappop(targets)
            target_columns.append(t[1])
            logging.debug("corr={}, y({})<=X({})".format(-t[0], t[1], ",".join(features)))
            error_attrs.remove(t[1])

        return target_columns

    def _compute_inference_order(self, env: Dict[str, str], train_df: DataFrame,
                                 error_attrs: List[str]) -> List[str]:
        # Defines a inference order based on `train_df`.
        #
        # TODO: Needs to analyze more dependencies (e.g., based on graph algorithms) between
        # target columns and the other ones for decideing a inference order.
        # For example, the SCARE paper [2] builds a dependency graph (a variant of graphical models)
        # to analyze the correlatioin of input data. But, the analysis is compute-intensive, so
        # we just use a naive approache to define the order now.
        #
        # TODO: Implements an `auto` strategy; it computes cost values by considering the three
        # existing strategies together and decides the inference order of attributes
        # based on the values.
        if self.inference_order == "domain":
            return self._domain_size_based_order(env, train_df, error_attrs)
        elif self.inference_order == "error":
            return self._error_num_based_order(error_attrs)

        assert self.inference_order == "entropy"
        return self._entropy_based_order(env, train_df, error_attrs)

    def _select_features(self, env: Dict[str, str], input_columns: List[str], y: pd.DataFrame,
                         excluded_columns: List[str]) -> List[str]:
        # All the available features
        features = [c for c in input_columns if c not in excluded_columns]

        # Selects features if necessary
        if self.max_training_column_num is not None and \
                int(self.max_training_column_num) < len(features):
            heap: List[Tuple[float, str]] = []
            for f, corr in map(lambda x: tuple(x), env["pairwise_attr_stats"][y]):
                if f in features:
                    # Converts to a negative value for extracting higher values
                    heapq.heappush(heap, (-float(corr), f))

            fts = [heapq.heappop(heap)[1] for i in range(int(self.max_training_column_num))]
            logging.debug("Select {} relevant features ({}) from available ones ({})".format(
                len(fts), ",".join(fts), ",".join(features)))
            features = fts

        return features

    def _transform_features(self, env: Dict[str, str], X: pd.DataFrame, features: List[str],
                            continous_attrs: List[str]) -> Tuple[pd.DataFrame, Any]:
        # Transforms discrete attributes with some categorical encoders if necessary
        import category_encoders as ce  # type: ignore[import]
        discrete_columns = [c for c in features if c not in continous_attrs]
        transformers = []
        if len(discrete_columns) != 0:
            # TODO: Needs to reconsider feature transformation in this part, e.g.,
            # we can use `ce.OrdinalEncoder` for small domain features. For the other category
            # encoders, see https://github.com/scikit-learn-contrib/category_encoders
            small_domain_columns = [
                c for c in discrete_columns
                if int(env["distinct_stats"][c]) < self.small_domain_threshold]  # type: ignore
            discrete_columns = [
                c for c in discrete_columns if c not in small_domain_columns]
            if len(small_domain_columns) > 0:
                transformers.append(ce.SumEncoder(
                    cols=small_domain_columns, handle_unknown='impute'))
            if len(discrete_columns) > 0:
                transformers.append(ce.OrdinalEncoder(
                    cols=discrete_columns, handle_unknown='impute'))
            # TODO: Needs to include `dirty_df` in this transformation
            for transformer in transformers:
                X = transformer.fit_transform(X)
            logging.debug("{} encoders transform ({})=>({})".format(
                len(transformers), ",".join(features), ",".join(X.columns)))

        # TODO: Even when using a GDBT, it might be better to standardize
        # continous values.

        return X, transformers

    @elapsed_time  # type: ignore
    def _build_model(self, X: pd.DataFrame, y: pd.Series, is_discrete: bool,
                     labels: List[str]) -> pd.DataFrame:
        import lightgbm as lgb  # type: ignore[import]

        # TODO: Validate given parameter values
        def _boosting_type() -> str:
            return self._get_option("lgb.boosting_type", "gbdt")

        def _class_weight() -> str:
            return self._get_option("lgb.class_weight", "balanced")

        def _learning_rate() -> float:
            return float(self._get_option("lgb.learning_rate", "0.01"))

        def _max_depth() -> int:
            return int(self._get_option("lgb.max_depth", "7"))

        def _max_bin() -> int:
            return int(self._get_option("lgb.max_bin", "255"))

        def _reg_alpha() -> float:
            return float(self._get_option("lgb.reg_alpha", "0.0"))

        def _min_split_gain() -> float:
            return float(self._get_option("lgb.min_split_gain", "0.0"))

        def _n_estimators() -> int:
            return int(self._get_option("lgb.n_estimators", "300"))

        def _n_splits() -> int:
            return int(self._get_option("cv.n_splits", "3"))

        def _max_eval() -> int:
            return int(self._get_option("hp.max_evals", "100000000"))

        def _no_progress_loss() -> int:
            return int(self._get_option("hp.no_progress_loss", "10"))

        if is_discrete:
            objective = "binary" if len(labels) <= 2 else "multiclass"
        else:
            objective = "regression"

        fixed_params = {
            "boosting_type": _boosting_type(),
            "objective": objective,
            "class_weight": _class_weight(),
            "learning_rate": _learning_rate(),
            "max_depth": _max_depth(),
            "max_bin": _max_bin(),
            "reg_alpha": _reg_alpha(),
            "min_split_gain": _min_split_gain(),
            "n_estimators": _n_estimators(),
            "random_state": 42,
            "n_jobs": -1
        }

        # Set `num_class` only in the `multiclass` mode
        if objective == "multiclass":
            fixed_params["num_class"] = len(labels)

        model_class = lgb.LGBMClassifier if is_discrete \
            else lgb.LGBMRegressor

        # If a task is clasification and #labels if less than 2,
        # we forcibly disable some optimizations.
        disable_opts = is_discrete and len(labels) < 2
        if disable_opts:
            logging.warn(
                f"Only {len(labels)} labels found, so forcibly disable learning optimizations "
                "(e.g., hyper param searches) when building a classifier")

        params: Dict[str, Any] = {}
        min_loss = float("nan")

        def _create_model(params: Dict[str, Any]) -> Any:
            # Some params must be int
            for k in ["num_leaves", "subsample_freq", "min_child_samples"]:
                if k in params:
                    params[k] = int(params[k])
            p = copy.deepcopy(fixed_params)
            p.update(params)
            return model_class(**p)

        if not disable_opts:
            from hyperopt import hp, tpe, Trials  # type: ignore[import]
            from hyperopt.early_stop import no_progress_loss  # type: ignore[import]
            from hyperopt.fmin import fmin  # type: ignore[import]
            from sklearn.model_selection import (  # type: ignore[import]
                cross_val_score, KFold, StratifiedKFold
            )

            # Forcibly disable INFO-level logging in the `hyperopt` module
            logger = logging.getLogger("hyperopt")
            logger.setLevel(logging.WARN)

            param_space = {
                "num_leaves": hp.quniform("num_leaves", 2, 100, 1),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "subsample_freq": hp.quniform("subsample_freq", 1, 20, 1),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.01, 1.0),
                "min_child_samples": hp.quniform("min_child_samples", 1, 50, 1),
                "min_child_weight": hp.loguniform("min_child_weight", -3, 1),
                "reg_lambda": hp.loguniform("reg_lambda", -2, 3)
            }

            scorer = "f1_macro" if is_discrete else "neg_mean_squared_error"
            cv = StratifiedKFold(n_splits=_n_splits(), shuffle=True) if is_discrete \
                else KFold(n_splits=_n_splits(), shuffle=True)

            # TODO: Columns where domain size is small are assumed to be categorical
            categorical_feature = "auto"

            def _objective(params: Dict[str, Any]) -> float:
                model = _create_model(params)
                fit_params = {
                    "categorical_feature": categorical_feature,
                    "verbose": 0
                }
                # TODO: Replace with `lgb.cv` to remove the `sklearn` dependency
                scores = cross_val_score(
                    model, X, y, scoring=scorer, cv=cv, fit_params=fit_params, n_jobs=-1)
                return -scores.mean()

            trials = Trials()

            best_params = fmin(
                fn=_objective,
                space=param_space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=_max_eval(),
                early_stop_fn=no_progress_loss(_no_progress_loss()),
                rstate=np.random.RandomState(42),
                show_progressbar=False,
                verbose=False)

            sorted_lst = sorted(trials.trials, key=lambda x: x['result']['loss'])
            min_loss = sorted_lst[0]['result']['loss']
            params = best_params

        # Builds a model with `best_params`
        model = _create_model(params)
        model.fit(X, y)

        return model, params, -min_loss

    @_spark_job_group(name="repair model training")
    def _build_repair_models(self, env: Dict[str, str], train_df: DataFrame, error_attrs: List[Row],
                             continous_attrs: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        # We now employ a simple repair model based on the SCARE paper [2] for scalable processing
        # on Apache Spark. Given a database tuple t = ce (c: correct attribute values,
        # e: error attribute values), the conditional probability of each combination of the
        # error attribute values c can be computed the product rule:
        #
        #  P(e\|c)=P(e[E_{1}]\|c)\prod_{i=2}^{K}P(e[E_{i}]\|c,e[E_{1}...E_{i-1}])
        #
        # where K is the number of error attributes, `len(error_attrs)`, and {E_[1], ..., E_[K]} is
        # a particular dependency order in error attributes. More sophisticated repair models
        # have been proposed recently, e.g., a Markov logic network based model in HoloClean [4].
        # Therefore, we might be able to improve our model more baesd on
        # the-state-of-the-art approaches.

        # Computes a inference order based on dependencies between `error_attrs` and the others
        target_columns = self._compute_inference_order(env, train_df, error_attrs)

        # Builds multiple ML models to repair error cells
        logging.info("[Repair Model Training Phase] Building {} ML models "
                     "to repair the cells in {} (order={})"
                     .format(len(target_columns), ",".join(target_columns), self.inference_order))

        if self.checkpoint_path is not None:
            # Keep a training table so that users can check later
            train_temp_view = f'train_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
            train_df.createOrReplaceTempView(train_temp_view)

            # Path to store models that will be built by the training process
            os.mkdir(self.checkpoint_path)

            logging.info(f"Model data checkpoint enabled for {train_temp_view}, "
                         f"the output path is '{self.checkpoint_path}'")

            with open(f"{self.checkpoint_path}/metadata.json", mode='w') as f:
                metadata = {
                    "train_table": train_temp_view,
                    "#rows": train_df.count(),
                    "sample_ratio": self.training_data_sample_ratio,
                    "columns": train_df.columns,
                    "inference_order": self.inference_order,
                    "target_columns": target_columns,
                    "pairwise_stats": env["pairwise_attr_stats"],
                    "distinct_stats": env["distinct_stats"]
                }
                json.dump(metadata, f, indent=2)

        models = {}
        train_pdf = train_df.toPandas()
        excluded_columns = copy.deepcopy(target_columns)
        for index, y in enumerate(target_columns):
            features = self._select_features(env, train_pdf.columns, y, excluded_columns)
            X, transformers = self._transform_features(
                env, train_pdf[features], features, continous_attrs)
            excluded_columns.remove(y)

            is_discrete = y not in continous_attrs
            model_type = "classifier" if is_discrete else "regressor"
            labels = train_df.selectExpr(f"collect_set(`{y}`) labels").collect()[0].labels \
                if is_discrete else []

            logging.info("Building {}/{} ML model... (type={}, y={}, features={}{})".format(
                index + 1, len(target_columns),
                model_type, y, ",".join(features),
                f", #labels={len(labels)}" if len(labels) > 0 else ""))
            (model, params, score), elapsed_time = \
                self._build_model(X, train_pdf[y], is_discrete, labels)
            logging.info("[{}/{}] score={} elapsed={}s".format(
                index + 1, len(target_columns), score, elapsed_time))

            models[y] = (model, features, transformers)

            if self.checkpoint_path is not None:
                checkpoint_prefix = f"{self.checkpoint_path}/{index}_{model_type}_{y}"
                # TODO: An error below happens when using `open` and `pickle` together:
                #   Argument 2 to "dump" has incompatible type "TextIO"; expected "IO[bytes]"
                # with open(f"{checkpoint_prefix}.pkl", mode='wb') as f:
                #     pickle.dump(model, f)
                try:
                    fd = open(f"{checkpoint_prefix}.pkl", mode='wb')
                    pickle.dump(model, fd)
                finally:
                    fd.close()
                with open(f"{checkpoint_prefix}.json", mode='w') as f:
                    metadata = {
                        "score": score,
                        "type": model_type,
                        "params": params,
                        "y": y,
                        "features": features,
                        "labels": labels
                    }
                    json.dump(metadata, f, indent=2)

        return models, target_columns

    @_spark_job_group(name="repairing")
    def _repair(self, env: Dict[str, str], models: Dict[str, Any], target_columns: List[str],
                continous_attrs: List[str], dirty_df: DataFrame, error_cells_df: DataFrame,
                compute_repair_candidate_prob: bool) -> pd.DataFrame:
        # Shares all the variables for the learnt models in a Spark cluster
        broadcasted_target_columns = self._spark.sparkContext.broadcast(target_columns)
        broadcasted_continous_attrs = self._spark.sparkContext.broadcast(continous_attrs)
        broadcasted_models = self._spark.sparkContext.broadcast(models)
        broadcasted_compute_repair_candidate_prob = \
            self._spark.sparkContext.broadcast(compute_repair_candidate_prob)
        broadcasted_maximal_likelihood_repair_enabled = \
            self._spark.sparkContext.broadcast(self.maximal_likelihood_repair_enabled)

        # Sets a grouping key for inference
        num_parallelism = self._spark.sparkContext.defaultParallelism
        # TODO: Fix this
        # grouping_key = self._temp_name("__grouping_key")
        grouping_key = "__grouping_key"
        env["dirty"] = self._create_temp_view(dirty_df, "dirty")
        dirty_df = dirty_df.withColumn(
            grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))

        # TODO: Run the `repair` UDF based on checkpoint files
        @functions.pandas_udf(dirty_df.schema, functions.PandasUDFType.GROUPED_MAP)
        def repair(pdf: pd.DataFrame) -> pd.DataFrame:
            target_columns = broadcasted_target_columns.value
            continous_attrs = broadcasted_continous_attrs.value
            models = broadcasted_models.value
            compute_repair_candidate_prob = broadcasted_compute_repair_candidate_prob.value
            maximal_likelihood_repair_enabled = \
                broadcasted_maximal_likelihood_repair_enabled.value
            rows: List[Row] = []
            for index, row in pdf.iterrows():
                for y in target_columns:
                    (model, features, transformers) = models[y]

                    # Preprocesses the input row for prediction
                    X = pd.DataFrame(row[features]).T
                    for c in [f for f in features if f in continous_attrs]:
                        X[c] = X[c].astype("float64")

                    # Transforms an input row to a feature
                    if transformers:
                        for transformer in transformers:
                            X = transformer.transform(X)

                    if y in continous_attrs:
                        if np.isnan(row[y]):
                            predicted = model.predict(X)
                            row[y] = float(predicted[0])
                    else:
                        if row[y] is None:
                            if compute_repair_candidate_prob or maximal_likelihood_repair_enabled:
                                predicted = model.predict_proba(X)
                                pmf = {"classes": model.classes_.tolist(),
                                       "probs": predicted[0].tolist()}
                                row[y] = json.dumps(pmf)
                            else:
                                predicted = model.predict(X)
                                row[y] = predicted[0]

                rows.append(row)

            return pd.DataFrame(rows)

        # Predicts the remaining error cells based on the trained models.
        # TODO: Might need to compare repair costs (cost of an update, c) to
        # the likelihood benefits of the updates (likelihood benefit of an update, l).
        logging.info(f"[Repairing Phase] Computing {error_cells_df.count()} repair updates in "
                     f"{dirty_df.count()} rows...")
        repaired_df = dirty_df.groupBy(grouping_key).apply(repair).drop(grouping_key).cache()
        repaired_df.write.format("noop").mode("overwrite").save()
        return repaired_df

    def _compute_repair_pmf(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        broadcasted_cf = self._spark.sparkContext.broadcast(self.distance)

        @functions.udf("array<double>")
        def cost_func(s1: str, s2: List[str]) -> List[float]:
            if s1 is not None:
                cf = broadcasted_cf.value
                return [cf.compute(s1, s) for s in s2]
            else:
                return [0.0] * len(s2)

        def _pmf_weight() -> float:
            return float(self._get_option("pfm.weight", "1.0"))

        parse_pmf_json_expr = "from_json(value, 'classes array<string>, probs array<double>') pmf"
        to_weighted_probs = f"zip_with(pmf.probs, costs, (p, c) -> p * (1.0 / (1.0 + {_pmf_weight()} * c))) probs"
        sum_probs = "aggregate(probs, double(0.0), (acc, x) -> acc + x) norm"
        to_pmf_expr = "arrays_zip(c, p) pmf"
        to_current_expr = "named_struct('value', current_value, 'prob', " \
            "coalesce(p[array_position(c, current_value) - 1], 0.0)) current"
        sorted_pmf_expr = "array_sort(pmf, (left, right) -> if(left.p < right.p, 1, -1)) pmf"
        pmf_df = self._flatten(self._create_temp_view(repaired_df)) \
            .join(error_cells_df, [self.row_id, "attribute"], "inner") \
            .selectExpr(self.row_id, "attribute", "current_value", parse_pmf_json_expr) \
            .withColumn("costs", cost_func(col("current_value"), col("pmf.classes"))) \
            .selectExpr(self.row_id, "attribute", "current_value", "pmf.classes classes", to_weighted_probs) \
            .selectExpr(self.row_id, "attribute", "current_value", "classes", "probs", sum_probs) \
            .selectExpr(self.row_id, "attribute", "current_value", "classes c", "transform(probs, p -> p / norm) p") \
            .selectExpr(self.row_id, "attribute", to_current_expr, to_pmf_expr) \
            .selectExpr(self.row_id, "attribute", "current", sorted_pmf_expr)

        return pmf_df

    def _compute_score(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        pmf_df = self._compute_repair_pmf(repaired_df, error_cells_df)

        broadcasted_distance = self._spark.sparkContext.broadcast(self.distance)

        @functions.pandas_udf("double")
        def distance(xs: pd.Series, ys: pd.Series) -> pd.Series:
            distance = broadcasted_distance.value
            dists = [distance.compute(x, y) for x, y in zip(xs, ys)]
            return pd.Series(dists)

        maximal_likelihood_repair_expr = "named_struct('value', pmf[0].c, 'prob', pmf[0].p) repaired"
        current_expr = "IF(ISNOTNULL(current.value), current.value, repaired.value)"
        score_expr = "ln(repaired.prob / IF(current.prob > 0.0, current.prob, 1e-6)) " \
            "* (1.0 / (1.0 + distance)) score"
        score_df = pmf_df \
            .selectExpr(self.row_id, "attribute", "current", maximal_likelihood_repair_expr) \
            .withColumn("distance", distance(expr(current_expr), col("repaired.value"))) \
            .selectExpr(self.row_id, "attribute", "current.value current_value",
                        "repaired.value repaired", score_expr)

        return score_df

    def _maximal_likelihood_repair(self, env: Dict[str, str], repaired_df: DataFrame,
                                   error_cells_df: DataFrame) -> DataFrame:
        # A “Maximal Likelihood Repair” problem defined in the SCARE [2] paper is as follows;
        # Given a scalar \delta and a database D = D_{e} \cup D_{c}, the problem is to
        # find another database instance D' = D'_{e} \cup D_{c} such that L(D'_{e} \| D_{c})
        # is maximum subject to the constraint Dist(D, D') <= \delta.
        # L is a likelihood function and Dist is an arbitrary distance function
        # (e.g., edit distances) between the two database instances D and D'.
        score_df = self._compute_score(repaired_df, error_cells_df)

        assert self.repair_delta is not None
        num_error_cells = error_cells_df.count()
        percent = min(1.0, 1.0 - self.repair_delta / num_error_cells)
        percentile = score_df.selectExpr(f"percentile(score, {percent}) thres").collect()[0]
        top_delta_repairs_df = score_df.where(f"score >= {percentile.thres}")
        logging.info("[Repairing Phase] {} repair updates (delta={}) selected "
                     "among {} candidates".format(
                         top_delta_repairs_df.count(),
                         self.repair_delta,
                         num_error_cells))

        return top_delta_repairs_df

    @elapsed_time  # type: ignore
    def _run(self, env: Dict[str, str], input_df: DataFrame, continous_attrs: List[str],
             detect_errors_only: bool, compute_training_target_hist: bool,
             compute_repair_candidate_prob: bool, compute_repair_prob: bool,
             compute_repair_score: bool, repair_data: bool) -> DataFrame:
        #################################################################################
        # 1. Error Detection Phase
        #################################################################################

        # If no error found, we don't need to do nothing
        gray_cells_df = self._detect_errors(env)
        if gray_cells_df.count() == 0:  # type: ignore
            logging.info("Any error cells not found, so the input data is already clean")
            return gray_cells_df if not repair_data else input_df

        # Sets NULL to suspicious cells
        repair_base_df = self._prepare_repair_base(env, gray_cells_df)

        # Selects error cells based on the result of domain analysis
        cell_domain_df = self._analyze_error_cell_domain(env, gray_cells_df, continous_attrs)

        # If `detect_errors_only` is True, returns found error cells
        error_cells_df = self._extract_error_cells(env, cell_domain_df, repair_base_df)
        if detect_errors_only:
            return error_cells_df

        # If no error found, we don't need to do nothing
        if error_cells_df.count() == 0:
            logging.info("Any error cells not found, so the input data is already clean")
            return error_cells_df if not repair_data else input_df

        #################################################################################
        # 2. Repair Model Training Phase
        #################################################################################

        # Selects rows for training, build models, and repair cells
        fixed_df, dirty_df, error_attrs = self._split_clean_and_dirty_rows(env, error_cells_df)
        if compute_training_target_hist:
            target_columns = list(map(lambda row: row.attribute, error_attrs))
            df = fixed_df.selectExpr(target_columns)
            hist_df = self._convert_to_histogram(df)
            # self._show_histogram(hist_df)
            return hist_df

        train_df = self._select_training_rows(fixed_df)
        min_training_row_num = int(float(env["num_input_rows"]) * self.min_training_row_ratio)
        if train_df.count() <= min_training_row_num:
            raise ValueError("Number of training rows must be greater than {} "
                             "(the {}%% number of input rows), but {} rows found".format(
                                 min_training_row_num,
                                 int(self.min_training_row_ratio * 100),
                                 train_df.count()))

        # Checks if we have the enough number of features for inference
        # TODO: In case of `num_features == 0`, we might be able to select the most accurate and
        # predictable column as a staring feature.
        num_features = len(train_df.columns) - len(error_attrs)
        if num_features == 0:
            raise ValueError("At least one feature is needed to repair error cells, "
                             "but no features found")

        models, target_columns = \
            self._build_repair_models(env, train_df, error_attrs, continous_attrs)

        #################################################################################
        # 3. Repair Phase
        #################################################################################

        repaired_df = self._repair(env, models, target_columns, continous_attrs,
                                   dirty_df, error_cells_df, compute_repair_candidate_prob)

        # If `compute_repair_candidate_prob` is True, returns probability mass function
        # of repair candidates.
        if compute_repair_candidate_prob:
            pmf_df = self._compute_repair_pmf(repaired_df, error_cells_df)
            # If `compute_repair_prob` is true, returns a predicted repair with
            # the highest probability only.
            if compute_repair_prob:
                return pmf_df.selectExpr(
                    self.row_id,
                    "attribute",
                    "current.value AS current_value",
                    "pmf[0].c AS repaired",
                    "pmf[0].p AS prob")
            elif compute_repair_score:
                return self._compute_score(repaired_df, error_cells_df)
            else:
                return pmf_df

        # If any discrete target columns and its probability distribution given,
        # computes scores to decide which cells should be repaired to follow the
        # “Maximal Likelihood Repair” problem.
        if self.maximal_likelihood_repair_enabled:
            top_delta_repairs_df = self._maximal_likelihood_repair(env, repaired_df, error_cells_df)
            if not repair_data:
                return top_delta_repairs_df

            # If `repair_data` is True, applys the selected repair updates into `dirty`
            env["top_delta_repairs"] = self._create_temp_view(
                top_delta_repairs_df, "top_delta_repairs")
            repaired_df = self._repair_attrs(env["top_delta_repairs"], env["dirty"])

        # If `repair_data` is False, returns repair candidates whoes
        # value is the same with `current_value`.
        if not repair_data:
            repair_candidates_df = self._flatten(self._create_temp_view(repaired_df)) \
                .join(error_cells_df, [self.row_id, "attribute"], "inner") \
                .selectExpr("tid", "attribute", "current_value", "value repaired") \
                .where("not(current_value <=> repaired)")
            return repair_candidates_df
        else:
            clean_df = fixed_df.union(repaired_df)
            assert clean_df.count() == input_df.count()
            return clean_df

    def run(self, detect_errors_only: bool = False, compute_repair_candidate_prob: bool = False,
            compute_repair_prob: bool = False, compute_repair_score: bool = False,
            compute_training_target_hist: bool = False,
            repair_data: bool = False) -> DataFrame:
        """
        Starts processing to detect error cells in given input data and build a statistical
        model to repair them.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        detect_errors_only : bool
            If set to ``True``, returns detected error cells (default: ``False``).
        compute_repair_candidate_prob : bool
            If set to ``True``, returns probabiity mass function of candidate
            repairs (default: ``False``).
        compute_repair_prob : bool
            If set to ``True``, returns probabiity of predicted repairs (default: ``False``).
        compute_training_target_hist: bool
            If set to ``True``, returns a histogram to analyze training data (default: ``False``).
        repair_data : bool
            If set to ``True``, returns repaired data (default: False).

        Examples
        --------
        >>> df = scavenger.repair.setInput(spark.table("adult")).setRowId("tid").run()
        >>> df.show()
        +---+---------+-------------+-----------+
        |tid|attribute|current_value|   repaired|
        +---+---------+-------------+-----------+
        | 12|      Age|         null|      18-21|
        | 12|      Sex|         null|     Female|
        |  7|      Sex|         null|     Female|
        |  3|      Sex|         null|     Female|
        |  5|      Age|         null|      18-21|
        |  5|   Income|         null|MoreThan50K|
        | 16|   Income|         null|MoreThan50K|
        +---+---------+-------------+-----------+

        >>> df = scavenger.repair.setInput(spark.table("adult")).setRowId("tid")
        ...    .run(compute_repair_prob=True)
        >>> df.show()
        +---+---------+-------------+-----------+-------------------+
        |tid|attribute|current_value|   repaired|               prob|
        +---+---------+-------------+-----------+-------------------+
        |  5|      Age|         null|      31-50| 0.5142776979219954|
        |  5|   Income|         null|LessThan50K| 0.9397100503416668|
        |  3|      Sex|         null|     Female| 0.6664498420338913|
        |  7|      Sex|         null|       Male| 0.7436767447201434|
        | 12|      Age|         null|        >50|0.40970902247819213|
        | 12|      Sex|         null|       Male| 0.7436767447201434|
        | 16|   Income|         null|LessThan50K| 0.9446392404617634|
        +---+---------+-------------+-----------+-------------------+
        """
        if self.input is None or self.row_id is None:
            raise ValueError("`setInput` and `setRowId` should be called before repairing")
        if self.maximal_likelihood_repair_enabled and self.repair_delta is None:
            raise ValueError("`setRepairDelta` should be called before "
                             "maximal likelihood repairing")
        if self.inference_order not in ["error", "domain", "entropy"]:
            raise ValueError(f"Inference order must be `error`, `domain`, or `entropy`, "
                             "but `{self.inference_order}` found")

        exclusive_param_list = [
            ("detect_errors_only", detect_errors_only),
            ("compute_repair_candidate_prob", compute_repair_candidate_prob),
            ("compute_repair_prob", compute_repair_prob),
            ("compute_repair_score", compute_repair_score),
            ("compute_training_target_hist", compute_training_target_hist),
            ("repair_data", repair_data)
        ]
        selected_param = list(map(lambda x: x[0], filter(lambda x: x[1], exclusive_param_list)))
        if len(selected_param) > 1:
            raise ValueError("{} cannot be set to True simultaneously".format(
                "/".join(map(lambda x: f"`{x}`", selected_param))))

        # To compute scores or the probabiity of predicted repairs, we need to compute
        # the probabiity mass function of candidate repairs.
        if compute_repair_prob or compute_repair_score:
            compute_repair_candidate_prob = True

        # A holder to keep runtime variables
        env: Dict[str, str] = {}

        try:
            # Validates input data
            input_df, continous_attrs = self._check_input_table(env)

            if compute_repair_candidate_prob and len(continous_attrs) != 0:
                raise ValueError("Cannot compute probability mass function of repairs "
                                 "when continous attributes found")
            if self.maximal_likelihood_repair_enabled and len(continous_attrs) != 0:
                raise ValueError("Cannot enable maximal likelihood repair mode "
                                 "when continous attributes found")

            df, elapsed_time = self._run(
                env, input_df, continous_attrs, detect_errors_only,
                compute_training_target_hist, compute_repair_candidate_prob,
                compute_repair_prob, compute_repair_score, repair_data)
            logging.info(f"!!!Total Processing time is {elapsed_time}(s)!!!")
            return df
        finally:
            self._release_resources()
