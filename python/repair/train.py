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
from typing import Any, Dict, List, Optional, Tuple

from repair.utils import elapsed_time, setup_logger


_logger = setup_logger()


# List of internal configurations
_opt_boosting_type = 'model.lgb.boosting_type'
_opt_class_weight = 'model.lgb.class_weight'
_opt_learning_rate = 'model.lgb.learning_rate'
_opt_max_depth = 'model.lgb.max_depth'
_opt_max_bin = 'model.lgb.max_bin'
_opt_reg_alpha = 'model.lgb.reg_alpha'
_opt_min_split_gain = 'model.lgb.min_split_gain'
_opt_n_estimators = 'model.lgb.n_estimators'
_opt_importance_type = 'model.lgb.importance_type'
_opt_n_splits = 'model.cv.n_splits'
_opt_timeout = 'model.hp.timeout'
_opt_max_evals = 'model.hp.max_evals'
_opt_no_progress_loss = 'model.hp.no_progress_loss'


@elapsed_time  # type: ignore
def _build_lgb_model(X: pd.DataFrame, y: pd.Series, is_discrete: bool, num_class: int, n_jobs: int,
                     opts: Dict[str, str]) -> Tuple[Any, float]:
    import lightgbm as lgb  # type: ignore[import]

    def _get_option_value(key: str, default_value: Any, type_class: Any = str) -> Any:
        assert type(default_value) is type_class
        if key in opts:
            try:
                return type_class(opts[key])
            except:
                _logger.warning(f'Failed to cast "{opts[key]}" into {type_class.__name__} data: key={key}')
                pass

        return default_value

    def _boosting_type() -> str:
        return _get_option_value(_opt_boosting_type, "gbdt")

    def _class_weight() -> str:
        return _get_option_value(_opt_class_weight, "balanced")

    def _learning_rate() -> float:
        return float(_get_option_value(_opt_learning_rate, 0.01, type_class=float))

    def _max_depth() -> int:
        return int(_get_option_value(_opt_max_depth, 7, type_class=int))

    def _max_bin() -> int:
        return int(_get_option_value(_opt_max_bin, 255, type_class=int))

    def _reg_alpha() -> float:
        return float(_get_option_value(_opt_reg_alpha, 0.0, type_class=float))

    def _min_split_gain() -> float:
        return float(_get_option_value(_opt_min_split_gain, 0.0, type_class=float))

    def _n_estimators() -> int:
        return int(_get_option_value(_opt_n_estimators, 300, type_class=int))

    def _importance_type() -> str:
        return _get_option_value(_opt_importance_type, "gain")

    def _n_splits() -> int:
        return int(_get_option_value(_opt_n_splits, 3, type_class=int))

    def _timeout() -> int:
        return int(_get_option_value(_opt_timeout, 0, type_class=int))

    def _max_eval() -> int:
        return int(_get_option_value(_opt_max_evals, 100000000, type_class=int))

    def _no_progress_loss() -> int:
        return int(_get_option_value(_opt_no_progress_loss, 50, type_class=int))

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
        "importance_type": _importance_type(),
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
    cv = StratifiedKFold(n_splits=_n_splits(), shuffle=True) if is_discrete \
        else KFold(n_splits=_n_splits(), shuffle=True)

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
        no_progress_loss_fn = no_progress_loss(_no_progress_loss())
        if _timeout() <= 0:
            return no_progress_loss_fn

        # Set base time for budget mechanism
        start_time = time.time()

        def timeout_fn(trials, best_loss=None, iteration_no_progress=0):  # type: ignore
            no_progress_loss, meta = no_progress_loss_fn(trials, best_loss, iteration_no_progress)
            timeout = time.time() - start_time > _timeout()
            return no_progress_loss or timeout, meta

        return timeout_fn

    try:
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

        _logger.info("hyperopt: #eval={}/{}".format(len(trials.trials), _max_eval()))

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


def model_configuration_keys() -> List[str]:
    return [
        _opt_boosting_type,
        _opt_class_weight,
        _opt_learning_rate,
        _opt_max_depth,
        _opt_max_bin,
        _opt_reg_alpha,
        _opt_min_split_gain,
        _opt_n_estimators,
        _opt_importance_type,
        _opt_n_splits,
        _opt_timeout,
        _opt_max_evals,
        _opt_no_progress_loss
    ]


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
