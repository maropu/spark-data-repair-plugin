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

from pyspark.sql import DataFrame, SparkSession, functions  # type: ignore[import]
from pyspark.sql.functions import col, expr  # type: ignore[import]

from repair.costs import UpdateCostFunction, NoCost
from repair.detectors import ConstraintErrorDetector, ErrorDetector, NullErrorDetector
from repair.utils import argtype_check, elapsed_time


class FunctionalDepModel():
    """
    Model class to mimic the scikit-learn APIs to predict values
    based on the rules of functional dependencies.

    .. versionchanged:: 0.1.0
    """

    def __init__(self, x: str, fd_map: Dict[str, str]) -> None:
        self.fd_map = fd_map
        self.classes = list(fd_map.values())
        self.x = x

        # Creates a var to map keys into their indexes on `fd_map.keys()`
        self.fd_keypos_map = {}
        for index, c in enumerate(self.classes):
            self.fd_keypos_map[c] = index

    @property
    def classes_(self) -> Any:
        return np.array(self.classes)

    def predict(self, X: pd.DataFrame) -> Any:
        return list(map(lambda x: self.fd_map[x] if x in self.fd_map else None, X[self.x]))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        pmf = []
        for x in X[self.x]:
            probs = np.zeros(len(self.classes))
            if x in self.fd_map.keys():
                probs[self.fd_keypos_map[self.fd_map[x]]] = 1.0
            pmf.append(probs)
        return pmf


class PoorModel():
    """Model to return the same value regardless of an input value.

    .. versionchanged:: 0.1.0
    """

    def __init__(self, v: Any) -> None:
        self.v = v

    @property
    def classes_(self) -> Any:
        return np.array([self.v])

    def predict(self, X: pd.DataFrame) -> Any:
        return [self.v] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return [np.array([1.0])] * len(X)


@elapsed_time  # type: ignore
def _build_lgb_model(X: pd.DataFrame, y: pd.Series, is_discrete: bool, num_class: int, n_jobs: int,
                     opts: Dict[str, str]) -> Any:
    import lightgbm as lgb  # type: ignore[import]

    # TODO: Validate given parameter values
    def _get_option(key: str, default_value: Optional[str]) -> Any:
        return opts[str(key)] if str(key) in opts else default_value

    def _boosting_type() -> str:
        return _get_option("lgb.boosting_type", "gbdt")

    def _class_weight() -> str:
        return _get_option("lgb.class_weight", "balanced")

    def _learning_rate() -> float:
        return float(_get_option("lgb.learning_rate", "0.01"))

    def _max_depth() -> int:
        return int(_get_option("lgb.max_depth", "7"))

    def _max_bin() -> int:
        return int(_get_option("lgb.max_bin", "255"))

    def _reg_alpha() -> float:
        return float(_get_option("lgb.reg_alpha", "0.0"))

    def _min_split_gain() -> float:
        return float(_get_option("lgb.min_split_gain", "0.0"))

    def _n_estimators() -> int:
        return int(_get_option("lgb.n_estimators", "300"))

    def _n_splits() -> int:
        return int(_get_option("cv.n_splits", "3"))

    def _parallelism() -> Optional[int]:
        opt_value = _get_option("hp.parallelism", None)
        return int(opt_value) if opt_value is not None else None

    def _timeout() -> Optional[int]:
        opt_value = _get_option("hp.timeout", None)
        return int(opt_value) if opt_value is not None else None

    def _max_eval() -> int:
        return int(_get_option("hp.max_evals", "100000000"))

    def _no_progress_loss() -> int:
        return int(_get_option("hp.no_progress_loss", "50"))

    if is_discrete:
        objective = "binary" if num_class <= 2 else "multiclass"
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
        "n_jobs": n_jobs
    }

    # Set `num_class` only in the `multiclass` mode
    if objective == "multiclass":
        fixed_params["num_class"] = num_class

    model_class = lgb.LGBMClassifier if is_discrete \
        else lgb.LGBMRegressor

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

    from hyperopt import hp, tpe, Trials  # type: ignore[import]
    from hyperopt.early_stop import no_progress_loss  # type: ignore[import]
    from hyperopt.fmin import fmin  # type: ignore[import]
    from sklearn.model_selection import (  # type: ignore[import]
        cross_val_score, KFold, StratifiedKFold
    )

    # TODO: Temporality supress `sklearn.model_selection` user's warning
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Forcibly disable INFO-level logging in the `hyperopt` module
    logging.getLogger("hyperopt").setLevel(logging.WARN)

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
            # TODO: Raises an error if a single regressor is used
            # "categorical_feature": categorical_feature,
            "verbose": 0
        }
        try:
            # TODO: Replace with `lgb.cv` to remove the `sklearn` dependency
            scores = cross_val_score(
                model, X, y, scoring=scorer, cv=cv, fit_params=fit_params, n_jobs=n_jobs)
            return -scores.mean()

        # it might throw an exception because `y` contains
        # previously unseen labels.
        except Exception as e:
            logging.warn(f"{e.__class__}: {e}")
            return 0.0

    def _early_stop_fn() -> Any:
        no_progress_loss_fn = no_progress_loss(_no_progress_loss())
        if _timeout() is None:
            return no_progress_loss_fn

        # Set base time for budget mechanism
        start_time = time.time()

        def timeout_fn(trials, best_loss=None, iteration_no_progress=0):  # type: ignore
            no_progress_loss, meta = no_progress_loss_fn(trials, best_loss, iteration_no_progress)
            timeout = time.time() - start_time > _timeout()
            return no_progress_loss or timeout, meta

        return timeout_fn

    trials = Trials()

    best_params = fmin(
        fn=_objective,
        space=param_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=_max_eval(),
        early_stop_fn=_early_stop_fn(),
        rstate=np.random.RandomState(42),
        show_progressbar=False,
        verbose=False)

    logging.info("hyperopt: #eval={}/{}".format(len(trials.trials), _max_eval()))

    sorted_lst = sorted(trials.trials, key=lambda x: x['result']['loss'])
    min_loss = sorted_lst[0]['result']['loss']
    params = best_params

    # Builds a model with `best_params`
    model = _create_model(params)
    model.fit(X, y)

    return model, params, -min_loss


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
        self.max_training_row_num: int = 10000
        self.max_training_column_num: Optional[int] = None
        self.small_domain_threshold: int = 12
        self.rule_based_model_enabled: bool = True
        self.parallel_stat_training_enabled: bool = False

        # Parameters for repairing
        self.maximal_likelihood_repair_enabled: bool = False
        self.repair_delta: Optional[int] = None

        # Defines a class to compute cost of updates.
        #
        # TODO: Needs a sophisticated way to compute update costs from a current value to a repair candidate.
        # For example, the HoloDetect paper [1] proposes a noisy channel model for the data augmentation
        # methodology of training data. This model consists of transformation rule and and data augmentation
        # policies (i.e., distribution over those data transformation).
        # This model might be able to represent this cost. For more details, see the section 5,
        # 'DATA AUGMENTATION LEARNING', in the paper.
        self.cf: UpdateCostFunction = NoCost()  # `NoCost` or `Levenshtein`

        # Options for internal behaviours
        self.opts: Dict[str, str] = {}

        # Temporary views to keep intermediate results; these views are automatically
        # created when repairing data, and then dropped finally.
        #
        # TODO: Move this variable into a runtime `env`
        self._intermediate_views_on_runtime: List[str] = []

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._jvm = self._spark.sparkContext._active_spark_context._jvm  # type: ignore
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
    def setErrorDetectors(self, detectors: List[Any]) -> "RepairModel":
        """
        Specifies the list of :class:`ErrorDetector` derived classes to implement
        a logic to detect error cells.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        detectors: list of :class:`ErrorDetector` derived classes
            specifies how to detect error cells. Available classes are as follows:

            * :class:`NullErrorDetector`: detects error cells based on NULL cells.
            * :class:`RegExErrorDetector`: detects error cells based on a regular expresson.
            * :class:`OutlierErrorDetector`: detects error cells based on the Gaussian distribution.
            * :class:`ConstraintErrorDetector`: detects error cells based on integrity rules
              defined by denial constraints.
        """
        # TODO: Removes this if `argtype_check` can handle this
        unknown_detectors = list(filter(lambda d: not isinstance(d, ErrorDetector), detectors))
        if len(unknown_detectors) > 0:
            raise TypeError("`detectors` should be provided as list[ErrorDetector], "
                            f"got {type(unknown_detectors[0]).__name__} in elements")
        self.error_detectors = detectors
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
    def setAttrMaxNumToComputeDomains(self, n: int) -> "RepairModel":
        """
        Specifies the max number of attributes to compute posterior probabiity
        based on the Naive Bayes assumption.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        n: int
            the max number of attributes (default: 4).
        """
        self.max_attrs_to_compute_domains = n
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
    def setMaxTrainingRowNum(self, n: int) -> "RepairModel":
        """
        Specifies the max number of training rows to build statistical models.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        n: int
            the max number of training data (default: 10000).
        """
        self.max_training_row_num = n
        return self

    @argtype_check  # type: ignore
    def setMaxTrainingColumnNum(self, n: int) -> "RepairModel":
        """
        Specifies the max number of training columns to build statistical models.

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
    def setRuleBasedModelEnabled(self, enabled: bool) -> "RepairModel":
        """Specifies whether to enable rule-based models based on functional dependencies.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        enabled: bool
            If set to ``True``, uses rule-based models if possible (default: ``False``).
        """
        self.rule_based_model_enabled = enabled
        return self

    @argtype_check  # type: ignore
    def setParallelStatTrainingEnabled(self, enabled: bool) -> "RepairModel":
        """Specifies whether to enable parallel training for stats repair models.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        enabled: bool
            If set to ``True``, runs multiples tasks to build stat repair models (default: ``False``).
        """
        self.parallel_stat_training_enabled = enabled
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
    def setUpdateCostFunction(self, cf: UpdateCostFunction) -> "RepairModel":
        """
        Specifies the :class:`UpdateCostFunction` derived class to implement
        a logic to compute update costs for repairs.

        .. versionchanged:: 0.1.0

        Parameters
        ----------
        cf: derived class of :class:`UpdateCostFunction`.
        """
        # TODO: Removes this if `argtype_check` can handle this
        if not isinstance(cf, UpdateCostFunction):
            raise TypeError("`cf` should be provided as UpdateCostFunction, "
                            f"got {type(cf)}")
        self.cf = cf
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
        if not all(c in df.columns for c in (str(self.row_id), "attribute")):  # type: ignore
            raise ValueError(f"Error cells must have `{self.row_id}` and "
                             "`attribute` in columns")
        return self._create_temp_view(df, "error_cells")

    def _get_option(self, key: str, default_value: Optional[str]) -> Any:
        return self.opts[str(key)] if str(key) in self.opts else default_value

    def _num_cores_per_executor(self) -> int:
        try:
            num_parallelism = self._spark.sparkContext.defaultParallelism
            num_executors = self._spark._jsc.sc().getExecutorMemoryStatus().size()  # type: ignore
            return max(1, num_parallelism / num_executors)
        except:
            return 1

    def _clear_job_group(self) -> None:
        # TODO: Uses `SparkContext.clearJobGroup()` instead
        self._spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)  # type: ignore
        self._spark.sparkContext.setLocalProperty("spark.job.description", None)  # type: ignore
        self._spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)  # type: ignore

    def _spark_job_group(name: str):  # type: ignore
        def decorator(f):  # type: ignore
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):  # type: ignore
                self._spark.sparkContext.setJobGroup(name, name)  # type: ignore
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

    def _create_temp_name(self, prefix: str = "temp") -> str:
        return f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    def _create_temp_view(self, df: Any, prefix: str = "temp") -> str:
        assert isinstance(df, DataFrame)
        temp_name = self._create_temp_name(prefix)
        df.createOrReplaceTempView(temp_name)
        self._intermediate_views_on_runtime.append(temp_name)
        return temp_name

    def _repair_attrs(self, repair_updates: str, base_table: str) -> DataFrame:
        jdf = self._jvm.RepairMiscApi.repairAttrsFrom(
            repair_updates, "", base_table, str(self.row_id))
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def _flatten(self, input_table: str) -> DataFrame:
        jdf = self._jvm.RepairMiscApi.flattenTable("", input_table, str(self.row_id))
        return DataFrame(jdf, self._spark._wrapped)  # type: ignore

    def _release_resources(self) -> None:
        while self._intermediate_views_on_runtime:
            v = self._intermediate_views_on_runtime.pop()
            logging.debug(f"Dropping an auto-generated view: {v}")
            self._spark.sql(f"DROP VIEW IF EXISTS {v}")

    def _check_input_table(self) -> Tuple[DataFrame, List[str], Dict[str, Any]]:
        ret_as_json = self._repair_api.checkInputTable(
            self.db_name, self._input_table, str(self.row_id))
        metadata = json.loads(ret_as_json)
        continous_attrs = metadata["continous_attrs"].split(",")
        return self._spark.table(metadata["input_table"]), \
            continous_attrs if continous_attrs != [""] else [], metadata

    def _detect_error_cells(self, input_table: str) -> DataFrame:
        # Initializes the given error detectors with the input params
        for d in self.error_detectors:
            d.setUp(str(self.row_id), input_table)  # type: ignore

        error_cells_dfs = [d.detect() for d in self.error_detectors]

        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        return err_cells_df.distinct().cache()

    @_spark_job_group(name="error detection")
    def _detect_errors(self, input_table: str, num_attrs: int, num_input_rows: int) -> DataFrame:
        # If `self.error_cells` provided, just uses it
        if self.error_cells is not None:
            # TODO: Even in this case, we need to use a NULL detector because
            # `_build_stat_model` will fail if `y` has NULL.
            noisy_cells_df = self._spark.table(self._error_cells)
            logging.info(f'[Error Detection Phase] Error cells provided by `{self._error_cells}`')

            # We assume that the given error cells are true, so we skip computing error domains
            # with probability because the computational cost is much high.
            self.domain_threshold_beta = 1.0
        else:
            # Applys error detectors to get noisy cells
            noisy_cells_df = self._detect_error_cells(input_table)
            logging.info(f'[Error Detection Phase] Detecting errors '
                         f'in a table `{input_table}` '
                         f'({num_attrs} cols x {num_input_rows} rows)...')

        # Filters target attributes if `self.targets` defined
        if len(self.targets) > 0:
            return noisy_cells_df.where("attribute IN ({})".format(",".join(map(lambda x: f"'{x}'", self.targets))))
        else:
            return noisy_cells_df

    def _prepare_repair_base(self, env: Dict[str, Any], noisy_cells_df: DataFrame) -> DataFrame:
        # Sets NULL at the detected noisy cells
        logging.debug("{}/{} noisy cells found, then converts them into NULL cells...".format(
            noisy_cells_df.count(), int(env["num_input_rows"]) * int(env["num_attrs"])))
        env.update(json.loads(self._repair_api.convertErrorCellsToNull(
            env["input_table"], env["noisy_cells"],
            str(self.row_id))))

        return self._register_and_get_df(env["repair_base"])

    def _preprocess(self, input_table: str, continous_attrs: List[str]) -> Tuple[str, str]:
        # Filters out attributes having large domains and makes continous values
        # discrete if necessary.
        ret_as_json = json.loads(self._repair_api.convertToDiscreteFeatures(
            input_table, str(self.row_id), self.discrete_thres))

        df = self._register_and_get_df(ret_as_json["discrete_features"])
        logging.debug("Valid {} attributes ({}) found in the {} input attributes ({}) and "
                      "{} continous attributes ({}) included in them".format(
                          len(df.columns),
                          ",".join(df.columns),
                          len(self._spark.table(input_table).columns),
                          ",".join(self._spark.table(input_table).columns),
                          len(continous_attrs),
                          ",".join(continous_attrs)))

        return ret_as_json["discrete_features"], ret_as_json["distinct_stats"]

    def _compute_attr_stats(self, discrete_features: str, noisy_cells: str) -> str:
        ret = json.loads(self._repair_api.computeAttrStats(
            discrete_features,
            noisy_cells,
            str(self.row_id),
            self.attr_stat_sample_ratio,
            self.attr_stat_threshold))
        self._intermediate_views_on_runtime.append(ret["attr_stats"])
        return ret["attr_stats"]

    @_spark_job_group(name="cell domain analysis")
    def _analyze_error_cell_domain(
            self, input_table: str, noisy_cells: str,
            continous_attrs: List[str]) -> Tuple[str, str, str]:

        # Checks if attributes are discrete or not, and discretizes continous ones
        discrete_features, distinct_stats = self._preprocess(input_table, continous_attrs)

        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        logging.debug("Collecting and sampling attribute stats (ratio={} threshold={}) "
                      "before computing error domains...".format(
                          self.attr_stat_sample_ratio,
                          self.attr_stat_threshold))

        attr_stats = self._compute_attr_stats(discrete_features, noisy_cells)

        logging.info("[Error Detection Phase] Analyzing cell domains to fix error cells...")
        ret_as_json = json.loads(self._repair_api.computeDomainInErrorCells(
            discrete_features, attr_stats, noisy_cells, str(self.row_id),
            ",".join(continous_attrs),
            self.max_attrs_to_compute_domains,
            self.min_corr_thres,
            self.domain_threshold_alpha,
            self.domain_threshold_beta))

        self._register_and_get_df(ret_as_json["cell_domain"])

        return ret_as_json["cell_domain"], distinct_stats, ret_as_json["pairwise_attr_stats"]

    def _extract_error_cells(self, env: Dict[str, Any], cell_domain_df: DataFrame,
                             repair_base_df: DataFrame) -> DataFrame:
        # Fixes cells if an inferred value is the same with an initial one
        fix_cells_expr = "if(current_value = domain[0].n, current_value, NULL) value"
        weak_df = cell_domain_df.selectExpr(
            str(self.row_id), "attribute", "current_value", fix_cells_expr).cache()
        error_cells_df = weak_df.where("value IS NULL").drop("value").cache()
        weak_df = weak_df.where("value IS NOT NULL") \
            .selectExpr(str(self.row_id), "attribute", "value repaired")
        env["weak"] = self._create_temp_view(weak_df.cache(), "weak")
        env["partial_repaired"] = self._create_temp_view(
            self._repair_attrs(env["weak"], env["repair_base"]),
            "partial_repaired")

        logging.info('[Error Detection Phase] {} noisy cells fixed and '
                     '{} error cells ({}%) remaining...'.format(
                         self._spark.table(env["weak"]).count(),
                         error_cells_df.count(),
                         error_cells_df.count() * 100.0 / (int(env["num_attrs"]) * int(env["num_input_rows"]))))

        return error_cells_df

    def _split_clean_and_dirty_rows(
            self, partial_repaired_df: DataFrame, error_cells_df: DataFrame) -> Tuple[DataFrame, DataFrame, List[str]]:
        error_rows_df = error_cells_df.selectExpr(str(self.row_id)).distinct().cache()
        fixed_rows_df = partial_repaired_df.join(error_rows_df, str(self.row_id), "left_anti").cache()
        dirty_rows_df = partial_repaired_df.join(error_rows_df, str(self.row_id), "left_semi").cache()
        error_attrs = error_cells_df.groupBy("attribute") \
            .agg(functions.count("attribute").alias("cnt")).collect()
        assert len(error_attrs) > 0
        return fixed_rows_df, dirty_rows_df, list(map(lambda x: x.attribute, error_attrs))

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

    # Selects relevant features if necessary. To reduce model training time,
    # it is important to drop non-relevant in advance.
    def _select_features(self, pairwise_stats: Dict[str, str], y: str, features: List[str]) -> List[str]:
        if self.max_training_column_num is not None and \
                int(self.max_training_column_num) < len(features):
            heap: List[Tuple[float, str]] = []
            for f, corr in map(lambda x: tuple(x), float(pairwise_stats[y])):  # type: ignore
                if f in features:
                    # Converts to a negative value for extracting higher values
                    heapq.heappush(heap, (-corr, f))

            fts = [heapq.heappop(heap) for i in range(len(features))]
            top_k_fts: List[Tuple[float, str]] = []
            for corr, f in fts:  # type: ignore
                # TODO: Parameterize a minimum corr to filter out irrelevant features
                if len(top_k_fts) <= 1 or (-corr >= 0.0 and len(top_k_fts) < int(self.max_training_column_num)):
                    top_k_fts.append((corr, f))

            logging.info("[Repair Model Training Phase] {} features ({}) selected from {} features".format(
                len(top_k_fts), ",".join(map(lambda f: f"{f[1]}:{-f[0]}", top_k_fts)), len(features)))

            features = list(map(lambda f: f[1], top_k_fts))

        return features

    def _create_transformers(self, distinct_stats: Dict[str, str], features: List[str],
                             continous_attrs: List[str]) -> List[Any]:
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
                if int(distinct_stats[c]) < self.small_domain_threshold]  # type: ignore
            discrete_columns = [
                c for c in discrete_columns if c not in small_domain_columns]
            if len(small_domain_columns) > 0:
                transformers.append(ce.SumEncoder(
                    cols=small_domain_columns, handle_unknown='impute'))
            if len(discrete_columns) > 0:
                transformers.append(ce.OrdinalEncoder(
                    cols=discrete_columns, handle_unknown='impute'))

        # TODO: Even when using a GDBT, it might be better to standardize
        # continous values.

        return transformers

    def _build_rule_model(self, train_df: DataFrame, target_columns: List[str], x: str, y: str) -> Any:
        # TODO: For attributes having large domain size, we need to rewrite it as a join query to repair data
        input_view = self._create_temp_view(train_df)
        ret_as_json = self._repair_api.computeFunctionDepMap(input_view, x, y)
        fd_map = json.loads(ret_as_json)
        return FunctionalDepModel(x, fd_map)

    def _get_functional_deps(self, train_df: DataFrame) -> Optional[Dict[str, List[str]]]:
        constraint_detectors = list(filter(lambda x: isinstance(x, ConstraintErrorDetector), self.error_detectors))
        # TODO: Supports the case where `self.error_detectors` has multiple `ConstraintErrorDetector`s
        if len(constraint_detectors) == 1:
            input_view = self._create_temp_view(train_df)
            constraint_path = constraint_detectors[0].constraint_path  # type: ignore
            ret_as_json = self._repair_api.computeFunctionalDeps(input_view, constraint_path)
            return json.loads(ret_as_json)
        else:
            return None

    def _build_repair_stat_models_in_series(
            self, models: Dict[str, Any], train_df: DataFrame,
            target_columns: List[str], continous_attrs: List[str],
            num_class_map: Dict[str, int],
            feature_map: Dict[str, List[str]],
            transformer_map: Dict[str, List[Any]]) -> Dict[str, Any]:

        if len(models) == len(target_columns):
            return models

        for y in target_columns:
            if y in models:
                continue

            index = len(models) + 1
            df = train_df.where(f"{y} IS NOT NULL")
            training_data_num = df.count()
            # Number of training data must be positive
            if training_data_num == 0:
                logging.info("Skipping {}/{} model... type=classfier y={} num_class={}".format(
                    index, len(target_columns), y, num_class_map[y]))
                models[y] = (PoorModel(None), feature_map[y], None)
                continue

            # The value of `max_training_row_num` highly depends on
            # the performance of pandas and LightGBM.
            sampling_ratio = float(self.max_training_row_num) / training_data_num \
                if training_data_num > self.max_training_row_num else 1.0

            # TODO: Needs more smart sampling, e.g., stratified sampling
            train_pdf = df.sample(sampling_ratio).toPandas()
            is_discrete = y not in continous_attrs

            X = train_pdf[train_pdf.columns[train_pdf.columns != y]]  # type: ignore
            for transformer in transformer_map[y]:
                X = transformer.fit_transform(X)
            logging.debug("{} encoders transform ({})=>({})".format(
                len(transformer_map[y]), ",".join(feature_map[y]), ",".join(X.columns)))

            logging.info("Building {}/{} model... type={} y={} features={} #rows={}{}".format(
                index, len(target_columns),
                "classfier" if y not in continous_attrs else "regressor",
                y, ",".join(feature_map[y]),
                len(train_pdf),
                f" #class={num_class_map[y]}" if num_class_map[y] > 0 else ""))
            ((model, params, score), elapsed_time) = \
                _build_lgb_model(X, train_pdf[y], is_discrete, num_class_map[y], n_jobs=-1, opts=self.opts)
            logging.info("Finishes building '{}' model...  score={} elapsed={}s".format(
                y, score, elapsed_time))

            models[y] = (model, feature_map[y], transformer_map[y])

        return models

    def _build_repair_stat_models_in_parallel(
            self, models: Dict[str, Any], train_df: DataFrame,
            target_columns: List[str], continous_attrs: List[str],
            num_class_map: Dict[str, int],
            feature_map: Dict[str, List[str]],
            transformer_map: Dict[str, List[Any]]) -> Dict[str, Any]:

        if len(models) == len(target_columns):
            return models

        # To build repair models in parallel, it assigns each model training into a single task
        train_dfs_per_target: List[DataFrame] = []
        target_column = self._create_temp_name("target_column")

        for y in target_columns:
            if y in models:
                continue

            index = len(models) + len(train_dfs_per_target) + 1
            df = train_df.where(f"{y} IS NOT NULL")
            training_data_num = df.count()
            # Number of training data must be positive
            if training_data_num == 0:
                logging.info("Skipping {}/{} model... type=classfier y={} num_class={}".format(
                    index, len(target_columns), y, num_class_map[y]))
                models[y] = (PoorModel(None), feature_map[y], None)
                continue

            # The value of `max_training_row_num` highly depends on
            # the performance of pandas and LightGBM.
            sampling_ratio = float(self.max_training_row_num) / training_data_num \
                if training_data_num > self.max_training_row_num else 1.0

            # TODO: Needs more smart sampling, e.g., stratified sampling
            df = df.sample(sampling_ratio)
            train_dfs_per_target.append(df.withColumn(target_column, functions.lit(y)))

            # TODO: Removes duplicate feature transformations
            train_pdf = df.toPandas()
            X = train_pdf[train_pdf.columns[train_pdf.columns != y]]  # type: ignore
            transformers = transformer_map[y]
            for transformer in transformers:
                X = transformer.fit_transform(X)
            logging.debug("{} encoders transform ({})=>({})".format(
                len(transformers), ",".join(feature_map[y]), ",".join(X.columns)))

            logging.info("Start building {}/{} model in parallel... type={} y={} features={} #rows={}{}".format(
                index, len(target_columns),
                "classfier" if y not in continous_attrs else "regressor",
                y, ",".join(feature_map[y]),
                len(train_pdf),
                f" #class={num_class_map[y]}" if num_class_map[y] > 0 else ""))

        num_tasks = len(train_dfs_per_target)
        if num_tasks == 0:
            return models

        broadcasted_target_column = self._spark.sparkContext.broadcast(target_column)
        broadcasted_continous_attrs = self._spark.sparkContext.broadcast(continous_attrs)
        broadcasted_transformer_map = self._spark.sparkContext.broadcast(transformer_map)
        broadcasted_num_class_map = self._spark.sparkContext.broadcast(num_class_map)
        broadcasted_n_jobs = self._spark.sparkContext.broadcast(max(1, int(self._num_cores_per_executor() / num_tasks)))
        broadcasted_opts = self._spark.sparkContext.broadcast(self.opts)

        @functions.pandas_udf("target: STRING, model: BINARY, score: DOUBLE, elapsed: DOUBLE",
                              functions.PandasUDFType.GROUPED_MAP)
        def train(pdf: pd.DataFrame) -> pd.DataFrame:
            target_column = broadcasted_target_column.value
            y = pdf.at[0, target_column]
            continous_attrs = broadcasted_continous_attrs.value
            transformers = broadcasted_transformer_map.value[y]
            is_discrete = y not in continous_attrs
            num_class = broadcasted_num_class_map.value[y]
            n_jobs = broadcasted_n_jobs.value
            opts = broadcasted_opts.value

            X = pdf.drop([y, target_column], axis=1)
            for transformer in transformers:
                X = transformer.transform(X)

            ((model, params, score), elapsed_time) = \
                _build_lgb_model(X, pdf[y], is_discrete, num_class, n_jobs, opts)
            row = [y, pickle.dumps(model), score, elapsed_time]
            return pd.DataFrame([row])

        # TODO: Any smart way to distribute tasks in different physical machines?
        built_models = functools.reduce(lambda x, y: x.union(y), train_dfs_per_target) \
            .groupBy(target_column).apply(train).collect()
        for row in built_models:
            logging.info("Finishes building '{}' model... score={} elapsed={}s".format(
                row.target, row.score, row.elapsed))

            model = pickle.loads(row.model)
            features = feature_map[row.target]
            transformers = transformer_map[row.target]
            models[row.target] = (model, features, transformers)

        return models

    @_spark_job_group(name="repair model training")
    def _build_repair_models(self, train_df: DataFrame, target_columns: List[str], continous_attrs: List[str],
                             distinct_stats: Dict[str, str],
                             pairwise_stats: Dict[str, str]) -> List[Any]:
        # We now employ a simple repair model based on the SCARE paper [2] for scalable processing
        # on Apache Spark. In the paper, given a database tuple t = ce (c: correct attribute values,
        # e: error attribute values), the conditional probability of each combination of the
        # error attribute values c can be computed using the product rule:
        #
        #  P(e\|c)=P(e[E_{1}]\|c)\prod_{i=2}^{|e|}P(e[E_{i}]\|c, r_{1}, ..., r_{i-1})
        #      , where r_{j} = if j = 1 then \arg\max_{e[E_{j}]} P(e[E_{j}]\|c)
        #                      else \arg\max_{e[E_{j}]} P(e[E_{j}]\|c, r_{1}, ..., r_{j-1})
        #
        # {E_{1}, ..., E_{|e|}} is an order to repair error attributes and it is determined by
        # a dependency graph of attributes. The SCARE repair model splits a database instance
        # into two parts: a subset D_{c} \subset D of clean (or correct) tuples and
        # D_{e} = D − D_{c} represents the remaining possibly dirty tuples.
        # Then, it trains the repair model P(e\|c) by using D_{c} and the model is used
        # to predict error attribute values in D_{e}.
        #
        # In our repair model, two minor improvements below are applied to enhance
        # precision and training speeds:
        #
        # - (1) Use NULL/weak-labeled cells for repair model training
        # - (2) Use functional dependency if possible
        #
        # In our model, we strongly assume error detectors can enumerate all the error cells,
        # that is, we can assume that non-blank cells are clean. Therefore, if c[x] -> e[y] in P(e[y]\|c)
        # and c[x] \in c (the value e[y] is determined by the value c[x]), we simply folow
        # this rule to skip expensive training costs.
        train_df = train_df.drop(str(self.row_id)).cache()

        # Selects features among input columns if necessary
        feature_map: Dict[str, List[str]] = {}
        transformer_map: Dict[str, List[Any]] = {}
        for y in target_columns:
            input_columns = [c for c in train_df.columns if c != y]  # type: ignore
            features = self._select_features(pairwise_stats, y, input_columns)  # type: ignore
            feature_map[y] = features
            transformer_map[y] = self._create_transformers(distinct_stats, features, continous_attrs)

        # If `self.rule_based_model_enabled` is `True`, try to analyze
        # functional deps on training data.
        functional_deps = self._get_functional_deps(train_df) \
            if self.rule_based_model_enabled else None
        if functional_deps is not None:
            logging.debug(f"Functional deps found: {functional_deps}")

        # Builds multiple repair models to repair error cells
        logging.info("[Repair Model Training Phase] Building {} models "
                     "to repair the cells in {}"
                     .format(len(target_columns), ",".join(target_columns)))

        models: Dict[str, Any] = {}
        num_class_map: Dict[str, int] = {}

        for y in target_columns:
            index = len(models) + 1
            is_discrete = y not in continous_attrs
            num_class_map[y] = train_df.selectExpr(f"count(distinct `{y}`) cnt").collect()[0].cnt \
                if is_discrete else 0

            # Skips building a model if num_class <= 1
            if is_discrete and num_class_map[y] <= 1:
                logging.info("Skipping {}/{} model... type=rule y={} num_class={}".format(
                    index, len(target_columns), y, num_class_map[y]))
                v = train_df.selectExpr(f"first(`{y}`) value").collect()[0].value \
                    if num_class_map[y] == 1 else None
                models[y] = (PoorModel(v), feature_map[y], None)

            # If `y` is functionally-dependent on one of clean attributes,
            # builds a model based on the rule.
            if y not in models and functional_deps is not None and y in functional_deps:
                def _max_domain_size() -> int:
                    return int(self._get_option("rule.max_domain_size", "1000"))

                def _qualified(x: str) -> bool:
                    # Checks if the domain size of `x` is small enough
                    return x in feature_map[y] and int(distinct_stats[x]) < _max_domain_size()

                fx = list(filter(lambda x: _qualified(x), functional_deps[y]))
                if len(fx) > 0:
                    logging.info("Building {}/{} model... type=rule(FD: X->y)  y={}(|y|={}) X={}(|X|={})".format(
                        index, len(target_columns), y, num_class_map[y], fx[0], distinct_stats[fx[0]]))
                    model = self._build_rule_model(train_df, target_columns, fx[0], y)
                    models[y] = (model, [fx[0]], None)

        build_stat_models = self._build_repair_stat_models_in_parallel \
            if self.parallel_stat_training_enabled else self._build_repair_stat_models_in_series
        build_stat_models(
            models, train_df, target_columns, continous_attrs,
            num_class_map, feature_map, transformer_map)

        assert len(models) == len(target_columns)

        # Resolve the conflict dependencies of the predictions
        if self.rule_based_model_enabled:
            import copy
            pred_ordered_models = []
            error_columns = copy.deepcopy(target_columns)

            # Appends no order-dependent models first
            for y in target_columns:
                (model, x, transformers) = models[y]
                if not isinstance(model, FunctionalDepModel):
                    pred_ordered_models.append((y, models[y]))
                    error_columns.remove(y)

            # Resolves an order for predictions
            while len(error_columns) > 0:
                columns = copy.deepcopy(error_columns)
                for y in columns:
                    (model, x, transformers) = models[y]
                    if x[0] not in error_columns:
                        pred_ordered_models.append((y, models[y]))
                        error_columns.remove(y)

                assert len(error_columns) < len(columns)

            logging.info("Resolved prediction order dependencies: {}".format(
                ",".join(map(lambda x: x[0], pred_ordered_models))))
            assert len(pred_ordered_models) == len(target_columns)
            return pred_ordered_models

        return list(models.items())

    @_spark_job_group(name="repairing")
    def _repair(self, models: List[Any], continous_attrs: List[str],
                dirty_rows_df: DataFrame, error_cells_df: DataFrame,
                compute_repair_candidate_prob: bool) -> pd.DataFrame:
        # Shares all the variables for the learnt models in a Spark cluster
        broadcasted_continous_attrs = self._spark.sparkContext.broadcast(continous_attrs)
        broadcasted_models = self._spark.sparkContext.broadcast(models)
        broadcasted_compute_repair_candidate_prob = \
            self._spark.sparkContext.broadcast(compute_repair_candidate_prob)
        broadcasted_maximal_likelihood_repair_enabled = \
            self._spark.sparkContext.broadcast(self.maximal_likelihood_repair_enabled)

        # Sets a grouping key for inference
        num_parallelism = self._spark.sparkContext.defaultParallelism
        grouping_key = self._create_temp_name("grouping_key")
        dirty_rows_df = dirty_rows_df.withColumn(
            grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))

        # TODO: Runs the `repair` UDF based on checkpoint files
        @functions.pandas_udf(dirty_rows_df.schema, functions.PandasUDFType.GROUPED_MAP)
        def repair(pdf: pd.DataFrame) -> pd.DataFrame:
            continous_attrs = broadcasted_continous_attrs.value
            models = broadcasted_models.value
            compute_repair_candidate_prob = broadcasted_compute_repair_candidate_prob.value
            maximal_likelihood_repair_enabled = \
                broadcasted_maximal_likelihood_repair_enabled.value

            for m in models:
                (y, (model, features, transformers)) = m

                # Preprocesses the input row for prediction
                X = pdf[features]

                # Transforms an input row to a feature
                if transformers:
                    for transformer in transformers:
                        X = transformer.transform(X)

                need_to_compute_pmf = y not in continous_attrs and \
                    (compute_repair_candidate_prob or maximal_likelihood_repair_enabled)
                if need_to_compute_pmf:
                    # TODO: Filters out top-k values to reduce the amount of data
                    predicted = model.predict_proba(X)
                    pmf = map(lambda p: {"classes": model.classes_.tolist(), "probs": p.tolist()}, predicted)
                    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
                    pdf[y] = pdf[y].where(pdf[y].notna(), list(pmf))
                else:
                    predicted = model.predict(X)
                    pdf[y] = pdf[y].where(pdf[y].notna(), predicted)

            return pdf

        # Predicts the remaining error cells based on the trained models.
        # TODO: Might need to compare repair costs (cost of an update, c) to
        # the likelihood benefits of the updates (likelihood benefit of an update, l).
        logging.info(f"[Repairing Phase] Computing {error_cells_df.count()} repair updates in "
                     f"{dirty_rows_df.count()} rows...")
        repaired_df = dirty_rows_df.groupBy(grouping_key).apply(repair).drop(grouping_key).cache()
        repaired_df.write.format("noop").mode("overwrite").save()
        return repaired_df

    def _compute_repair_pmf(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        broadcasted_cf = self._spark.sparkContext.broadcast(self.cf)

        @functions.udf("array<double>")
        def cost_func(s1: str, s2: List[str]) -> List[float]:
            if s1 is not None:
                cf = broadcasted_cf.value
                return [cf.compute(s1, s) for s in s2]
            else:
                return [0.0] * len(s2)

        def _pmf_weight() -> float:
            return float(self._get_option("pmf.cost_weight", "0.1"))

        parse_pmf_json_expr = "from_json(value, 'classes array<string>, probs array<double>') pmf"
        slice_probs = "slice(pmf.probs, 1, size(pmf.classes)) probs"
        to_weighted_probs = f"zip_with(probs, costs, (p, c) -> p * (1.0 / (1.0 + {_pmf_weight()} * c))) probs"
        sum_probs = "aggregate(probs, double(0.0), (acc, x) -> acc + x) norm"
        to_pmf_expr = "filter(arrays_zip(c, p), x -> x.p > 0.01) pmf"
        normalize_probs = "transform(probs, p -> p / norm) p"
        to_current_expr = "named_struct('value', current_value, 'prob', " \
            "coalesce(p[array_position(c, current_value) - 1], 0.0)) current"
        sorted_pmf_expr = "array_sort(pmf, (left, right) -> if(left.p < right.p, 1, -1)) pmf"
        pmf_df = self._flatten(self._create_temp_view(repaired_df)) \
            .join(error_cells_df, [str(self.row_id), "attribute"], "inner") \
            .selectExpr(str(self.row_id), "attribute", "current_value", parse_pmf_json_expr) \
            .selectExpr(str(self.row_id), "attribute", "current_value", "pmf.classes classes", slice_probs) \
            .withColumn("costs", cost_func(col("current_value"), col("classes"))) \
            .selectExpr(str(self.row_id), "attribute", "current_value", "classes", to_weighted_probs) \
            .selectExpr(str(self.row_id), "attribute", "current_value", "classes", "probs", sum_probs) \
            .selectExpr(str(self.row_id), "attribute", "current_value", "classes c", normalize_probs) \
            .selectExpr(str(self.row_id), "attribute", to_current_expr, to_pmf_expr) \
            .selectExpr(str(self.row_id), "attribute", "current", sorted_pmf_expr)

        return pmf_df

    def _compute_score(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        pmf_df = self._compute_repair_pmf(repaired_df, error_cells_df)

        broadcasted_cf = self._spark.sparkContext.broadcast(self.cf)

        @functions.pandas_udf("double")  # type: ignore
        def cost_func(xs: pd.Series, ys: pd.Series) -> pd.Series:
            cf = broadcasted_cf.value
            dists = [cf.compute(x, y) for x, y in zip(xs, ys)]
            return pd.Series(dists)

        maximal_likelihood_repair_expr = "named_struct('value', pmf[0].c, 'prob', pmf[0].p) repaired"
        current_expr = "IF(ISNOTNULL(current.value), current.value, repaired.value)"
        score_expr = "ln(repaired.prob / IF(current.prob > 0.0, current.prob, 1e-6)) " \
            "* (1.0 / (1.0 + cost)) score"
        score_df = pmf_df \
            .selectExpr(str(self.row_id), "attribute", "current", maximal_likelihood_repair_expr) \
            .withColumn("cost", cost_func(expr(current_expr), col("repaired.value"))) \
            .selectExpr(str(self.row_id), "attribute", "current.value current_value",
                        "repaired.value repaired", score_expr)

        return score_df

    def _maximal_likelihood_repair(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        # A “Maximal Likelihood Repair” problem defined in the SCARE [2] paper is as follows;
        # Given a scalar \delta and a database D = D_{e} \cup D_{c}, the problem is to
        # find another database instance D' = D'_{e} \cup D_{c} such that L(D'_{e} \| D_{c})
        # is maximum subject to the constraint Cost(D, D') <= \delta.
        # L is a likelihood function and Cost is an arbitrary update cost function
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
    def _run(self, env: Dict[str, Any], input_df: DataFrame, continous_attrs: List[str],
             detect_errors_only: bool, compute_training_target_hist: bool,
             compute_repair_candidate_prob: bool, compute_repair_prob: bool,
             compute_repair_score: bool, repair_data: bool) -> DataFrame:
        #################################################################################
        # 1. Error Detection Phase
        #################################################################################

        # If no error found, we don't need to do nothing
        noisy_cells_df = self._detect_errors(
            env["input_table"], env["num_attrs"], env["num_input_rows"])
        # TODO: Remove this
        env["noisy_cells"] = self._create_temp_view(noisy_cells_df, "noisy_cells")
        if noisy_cells_df.count() == 0:  # type: ignore
            logging.info("Any error cells not found, so the input data is already clean")
            return noisy_cells_df if not repair_data else input_df

        # Sets NULL to noisy cells
        repair_base_df = self._prepare_repair_base(env, noisy_cells_df)

        # Selects error cells based on the result of domain analysis
        cell_domain, distinct_stats, pairwise_stats = self._analyze_error_cell_domain(
            env["input_table"], env["noisy_cells"], continous_attrs)

        # If `detect_errors_only` is True, returns found error cells
        cell_domain_df = self._spark.table(cell_domain)
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
        fixed_rows_df, dirty_rows_df, target_columns = self._split_clean_and_dirty_rows(
            self._spark.table(env["partial_repaired"]), error_cells_df)
        if compute_training_target_hist:
            df = fixed_rows_df.selectExpr(target_columns)
            hist_df = self._convert_to_histogram(df)
            # self._show_histogram(hist_df)
            return hist_df

        # Checks if we have the enough number of features for inference
        # TODO: In case of `num_features == 0`, we might be able to select the most accurate and
        # predictable column as a staring feature.
        num_features = len(fixed_rows_df.columns) - len(target_columns)
        if num_features == 0:
            raise ValueError("At least one feature is needed to repair error cells, "
                             "but no features found")

        train_df = self._spark.table(env["partial_repaired"])
        models = self._build_repair_models(
            train_df, target_columns, continous_attrs, distinct_stats, pairwise_stats)

        #################################################################################
        # 3. Repair Phase
        #################################################################################

        repaired_df = self._repair(
            models, continous_attrs, dirty_rows_df, error_cells_df,
            compute_repair_candidate_prob)

        # If `compute_repair_candidate_prob` is True, returns probability mass function
        # of repair candidates.
        if compute_repair_candidate_prob:
            pmf_df = self._compute_repair_pmf(repaired_df, error_cells_df)
            # If `compute_repair_prob` is true, returns a predicted repair with
            # the highest probability only.
            if compute_repair_prob:
                return pmf_df.selectExpr(
                    str(self.row_id),
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
            top_delta_repairs_df = self._maximal_likelihood_repair(repaired_df, error_cells_df)
            if not repair_data:
                return top_delta_repairs_df

            # If `repair_data` is True, applys the selected repair updates into `dirty_rows`
            top_delta_repairs = self._create_temp_view(top_delta_repairs_df, "top_delta_repairs")
            dirty_rows = self._create_temp_view(dirty_rows_df, "dirty_rows")
            repaired_df = self._repair_attrs(top_delta_repairs, dirty_rows)

        # If `repair_data` is False, returns repair candidates whoes
        # value is the same with `current_value`.
        if not repair_data:
            repair_candidates_df = self._flatten(self._create_temp_view(repaired_df)) \
                .join(error_cells_df, [str(self.row_id), "attribute"], "inner") \
                .selectExpr("tid", "attribute", "current_value", "value repaired") \
                .where("repaired IS NULL OR not(current_value <=> repaired)")
            return repair_candidates_df
        else:
            clean_df = fixed_rows_df.union(repaired_df)
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
        env: Dict[str, Any] = {}

        try:
            # Validates input data
            input_df, continous_attrs, metadata = self._check_input_table()
            env.update(metadata)

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
