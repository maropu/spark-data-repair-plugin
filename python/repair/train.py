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
import time
import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

from repair.utils import elapsed_time, get_option_value, setup_logger


_logger = setup_logger()


# List of internal configurations
_option = namedtuple('_option', 'key default_value type_class validator err_msg')

_opt_boosting_type = \
    _option('model.lgb.boosting_type', 'gbdt', str,
            lambda v: v in ['gbdt', 'dart', 'goss', 'rf'], "`{}` should be in ['gbdt', 'dart', 'goss', 'rf']")
_opt_class_weight = \
    _option('model.lgb.class_weight', 'balanced', str, None, None)
_opt_learning_rate = \
    _option('model.lgb.learning_rate', 0.01, float,
            lambda v: v > 0.0, '`{}` should be positive')
_opt_max_depth = \
    _option('model.lgb.max_depth', 7, int, None, None)
_opt_max_bin = \
    _option('model.lgb.max_bin', 255, int, None, None)
_opt_reg_alpha = \
    _option('model.lgb.reg_alpha', 0.0, float,
            lambda v: v >= 0.0, '`{}` should be greater than or equal to 0.0')
_opt_min_split_gain = \
    _option('model.lgb.min_split_gain', 0.0, float,
            lambda v: v >= 0.0, '`{}` should be greater than or equal to 0.0')
_opt_n_estimators = \
    _option('model.lgb.n_estimators', 300, int,
            lambda v: v > 0, '`{}` should be positive')
_opt_importance_type = \
    _option('model.lgb.importance_type', 'gain', str,
            lambda v: v in ['split', 'gain'], "`{}` should be in ['split', 'gain']")
_opt_n_splits = \
    _option('model.cv.n_splits', 3, int,
            lambda v: v >= 3, '`{}` should be greater than 2')
_opt_timeout = \
    _option('model.hp.timeout', 0, int, None, None)
_opt_max_evals = \
    _option('model.hp.max_evals', 100000000, int,
            lambda v: v > 0, '`{}` should be positive')
_opt_no_progress_loss = \
    _option('model.hp.no_progress_loss', 50, int,
            lambda v: v > 0, '`{}` should be positive')

train_option_keys = [
    _opt_boosting_type.key,
    _opt_class_weight.key,
    _opt_learning_rate.key,
    _opt_max_depth.key,
    _opt_max_bin.key,
    _opt_reg_alpha.key,
    _opt_min_split_gain.key,
    _opt_n_estimators.key,
    _opt_importance_type.key,
    _opt_n_splits.key,
    _opt_timeout.key,
    _opt_max_evals.key,
    _opt_no_progress_loss.key
]


@elapsed_time  # type: ignore
def _build_lgb_model(X: pd.DataFrame, y: pd.Series, is_discrete: bool, num_class: int, n_jobs: int,
                     opts: Dict[str, str]) -> Tuple[Any, float]:
    import lightgbm as lgb  # type: ignore[import]

    def _get_option_value(*args) -> Any:  # type: ignore
        return get_option_value(opts, *args)

    if is_discrete:
        objective = "binary" if num_class <= 2 else "multiclass"
    else:
        objective = "regression"

    fixed_params = {
        "boosting_type": _get_option_value(*_opt_boosting_type),
        "objective": objective,
        "class_weight": _get_option_value(*_opt_class_weight),
        "learning_rate": _get_option_value(*_opt_learning_rate),
        "max_depth": _get_option_value(*_opt_max_depth),
        "max_bin": _get_option_value(*_opt_max_bin),
        "reg_alpha": _get_option_value(*_opt_reg_alpha),
        "min_split_gain": _get_option_value(*_opt_min_split_gain),
        "n_estimators": _get_option_value(*_opt_n_estimators),
        "importance_type": _get_option_value(*_opt_importance_type),
        "random_state": 42,
        "n_jobs": n_jobs
    }

    # Set `num_class` only in the `multiclass` mode
    if objective == "multiclass":
        fixed_params["num_class"] = num_class

    model_class = lgb.LGBMClassifier if is_discrete \
        else lgb.LGBMRegressor

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
    from logging import getLogger, WARN
    getLogger("hyperopt").setLevel(WARN)

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
    n_splits = int(_get_option_value(*_opt_n_splits))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True) if is_discrete \
        else KFold(n_splits=n_splits, shuffle=True)

    def _objective(params: Dict[str, Any]) -> float:
        model = _create_model(params)
        fit_params: Dict[str, str] = {
            # TODO: Raises an error if a single regressor is used
            # "categorical_feature": "auto",
        }
        try:
            # TODO: Replace with `lgb.cv` to remove the `sklearn` dependency
            scores = cross_val_score(
                model, X, y, scoring=scorer, cv=cv, fit_params=fit_params, n_jobs=n_jobs)
            return -scores.mean()

        # it might throw an exception because `y` contains
        # previously unseen labels.
        except Exception as e:
            _logger.warning(f"{e.__class__}: {e}")
            return 0.0

    def _early_stop_fn() -> Any:
        no_progress_loss_fn = no_progress_loss(int(_get_option_value(*_opt_no_progress_loss)))
        timeout = int(_get_option_value(*_opt_timeout))
        if timeout <= 0:
            return no_progress_loss_fn

        # Set base time for budget mechanism
        start_time = time.time()

        def timeout_fn(trials, best_loss=None, iteration_no_progress=0):  # type: ignore
            no_progress_loss, meta = no_progress_loss_fn(trials, best_loss, iteration_no_progress)
            to = time.time() - start_time > timeout
            return no_progress_loss or to, meta

        return timeout_fn

    try:
        trials = Trials()
        max_evals = int(_get_option_value(*_opt_max_evals))
        best_params = fmin(
            fn=_objective,
            space=param_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=max_evals,
            early_stop_fn=_early_stop_fn(),
            rstate=np.random.RandomState(42),
            show_progressbar=False,
            verbose=False)

        _logger.info("hyperopt: #eval={}/{}".format(len(trials.trials), max_evals))

        # Builds a model with `best_params`
        # TODO: Could we extract constraint rules (e.g., FD and CFD) from built statistical models?
        model = _create_model(best_params)
        model.fit(X, y)

        def _feature_importances() -> List[Any]:
            f = filter(lambda x: x[1] > 0.0, zip(model.feature_name_, model.feature_importances_))
            return list(sorted(f, key=lambda x: x[1], reverse=True))

        _logger.debug(f"lightgbm: feature_importances={_feature_importances()}")

        sorted_lst = sorted(trials.trials, key=lambda x: x['result']['loss'])
        min_loss = sorted_lst[0]['result']['loss']
        return model, -min_loss
    except Exception as e:
        _logger.warning(f"Failed to build a stat model because: {e}")
        return None, 0.0


def build_model(X: pd.DataFrame, y: pd.Series, is_discrete: bool, num_class: int, n_jobs: int,
                opts: Dict[str, str]) -> Tuple[Any, float]:
    return _build_lgb_model(X, y, is_discrete, num_class, n_jobs, opts)


def compute_class_nrow_stdv(y: pd.Series, is_discrete: bool) -> Optional[float]:
    from collections import Counter
    return float(np.std(list(map(lambda x: x[1], Counter(y).items())))) if is_discrete else None


def rebalance_training_data(X: pd.DataFrame, y: pd.Series, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Uses median as the number of training rows for each class
    from collections import Counter
    prev_nrows = len(X)
    prev_stdv = compute_class_nrow_stdv(y, is_discrete=True)
    hist = dict(Counter(y).items())  # type: ignore
    median = int(np.median([count for key, count in hist.items()]))

    def _split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[df.columns[df.columns != target]]  # type: ignore
        y = df[target]
        return X, y

    # Filters out rows having NaN values for over-sampling
    X[target] = y
    X_notna, y_notna = _split_data(X.dropna())
    X_na, y_na = _split_data(X[X.isnull().any(axis=1)])

    # Over-sampling for training data whose row number is smaller than the median value
    hist_na = dict(Counter(y_na).items())  # type: ignore
    smote_targets = []
    kn = 5  # `k_neighbors` default value in `SMOTEN`
    for key, count in hist.items():
        if count < median:
            nna = hist_na[key] if key in hist_na else 0
            if count - nna > kn:
                smote_targets.append((key, median - nna))
            else:
                _logger.warning(f"Over-sampling of '{key}' in y='{target}' failed because the number of the clean rows "
                                f"is too small: {count - nna}")

    if len(smote_targets) > 0:
        from imblearn.over_sampling import SMOTEN
        sampler = SMOTEN(random_state=42, sampling_strategy=dict(smote_targets), k_neighbors=kn)
        X_notna, y_notna = sampler.fit_resample(X_notna, y_notna)

    X = pd.concat([X_notna, X_na])
    y = pd.concat([y_notna, y_na])

    # Under-sampling for training data whose row number is greater than the median value
    rus_targets = list(map(lambda x: (x[0], median), filter(lambda x: x[1] > median, hist.items())))
    if len(rus_targets) > 0:
        # NOTE: The other smarter implementations can skew samples if there are many rows having NaN values,
        # so we just use `RandomUnderSampler` here.
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=dict(rus_targets))
        X, y = sampler.fit_resample(X, y)

    _logger.info("Rebalanced training data (y={}, median={}): #rows={}(stdv={}) -> #rows={}(stdv={})".format(
        target, median, prev_nrows, prev_stdv, len(X), compute_class_nrow_stdv(y, is_discrete=True)))
    _logger.debug("class hist: {} => {}".format(hist.items(), Counter(y).items()))
    return X, y
