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
import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, Row, SparkSession, functions
from pyspark.sql.functions import col

from repair.detectors import ErrorDetector, NullErrorDetector
from repair.distances import Distance, Levenshtein


class RepairModel():

    def __init__(self) -> None:
        super().__init__()

        # Basic parameters
        self.db_name: str = ""
        self.table_name: Optional[str] = None
        self.row_id: Optional[str] = None

        # Parameters for error detection
        self.error_cells: Optional[str] = None
        # To find error cells, the NULL detector is used by default
        self.error_detectors: List[ErrorDetector] = [NullErrorDetector()]
        self.discrete_thres: float = 80
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
        self.lgb_num_leaves: int = 31
        self.lgb_max_depth: int = -1

        # Parameters for repairing
        self.repair_updates: Optional[str] = None
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

        # Internally used to check elapsed time
        self._timer_base: Optional[float] = None

        # Temporary views used in repairing processes
        # TODO: Creates a temp database for these views then drop it finally
        self._meta_view_names: List[str] = [
            "discrete_features",
            "gray_cells",
            "repair_base",
            "cell_domain",
            "partial_repaired",
            "repaired",
            "dirty",
            "weak",
            "train",
            "top_delta_repairs"
        ]

        # JVM interfaces for Data Repair APIs
        self._spark = SparkSession.builder.getOrCreate()
        self._jvm = self._spark.sparkContext._active_spark_context._jvm
        self._svg_api = self._jvm.RepairApi

    def setDbName(self, db_name: str) -> "RepairModel":
        self.db_name = db_name
        return self

    def setTableName(self, table_name: str) -> "RepairModel":
        self.table_name = table_name
        return self

    # TODO: Add `def setDf(self, df: DataFrame) -> "RepairModel"`

    def setRowId(self, row_id: str) -> "RepairModel":
        self.row_id = row_id
        return self

    def setErrorCells(self, error_cells: str) -> "RepairModel":
        self.error_cells = error_cells
        return self

    def setErrorDetector(self, detector: ErrorDetector) -> "RepairModel":
        if not isinstance(detector, ErrorDetector):
            raise ValueError("Error detector must derive a base class "
                             "`repair.detectors.ErrorDetector`")
        self.error_detectors.append(detector)
        return self

    def setDiscreteThreshold(self, thres: float) -> "RepairModel":
        self.discrete_thres = thres
        return self

    def setMinCorrThreshold(self, thres: float) -> "RepairModel":
        self.min_corr_thres = thres
        return self

    def setDomainThresholds(self, alpha: float, beta: float) -> "RepairModel":
        self.domain_threshold_alpha = alpha
        self.domain_threshold_beta = beta
        return self

    def setAttrMaxNumToComputeDomains(self, max: int) -> "RepairModel":
        self.max_attrs_to_compute_domains = max
        return self

    def setAttrStatSampleRatio(self, ratio: float) -> "RepairModel":
        self.attr_stat_sample_ratio = ratio
        return self

    def setAttrStatThreshold(self, ratio: float) -> "RepairModel":
        self.attr_stat_threshold = ratio
        return self

    def setTrainingDataSampleRatio(self, ratio: float) -> "RepairModel":
        self.training_data_sample_ratio = ratio
        return self

    def setMaxTrainingColumnNum(self, n: int) -> "RepairModel":
        self.max_training_column_num = n
        return self

    def setSmallDomainThreshold(self, thres: int) -> "RepairModel":
        self.small_domain_threshold = thres
        return self

    def setInferenceOrder(self, inference_order: str) -> "RepairModel":
        self.inference_order = inference_order
        return self

    def setLGBNumLeaves(self, n: int) -> "RepairModel":
        self.lgb_num_leaves = n
        return self

    def setLGBMaxDepth(self, n: int) -> "RepairModel":
        self.lgb_max_depth = n
        return self

    def setRepairUpdates(self, repair_updates: str) -> "RepairModel":
        self.repair_updates = repair_updates
        return self

    def setMaximalLikelihoodRepairEnabled(self, enabled: bool) -> "RepairModel":
        self.maximal_likelihood_repair_enabled = enabled
        return self

    def setRepairDelta(self, delta: int) -> "RepairModel":
        if delta <= 0:
            raise ValueError("Repair delta must be positive")
        self.repair_delta = delta
        return self

    def setDistance(self, distance: Distance) -> "RepairModel":
        if not isinstance(distance, Distance):
            raise ValueError("Distance function must derive a base class "
                             "`repair.distances.Distance`")
        self.distance = distance
        return self

    def _temp_name(self, prefix: str = "temp") -> str:
        return f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    def _temp_view(self, df: DataFrame) -> str:
        temp_view = self._temp_name()
        df.createOrReplaceTempView(temp_view)
        return temp_view

    def _flatten(self, df: DataFrame) -> DataFrame:
        temp_view = self._temp_name()
        df.createOrReplaceTempView(temp_view)
        jdf = self._jvm.RepairMiscApi.flattenTable("", temp_view, self.row_id)
        return DataFrame(jdf, self._spark._wrapped)

    def _start_spark_jobs(self, name: str, desc: str) -> None:
        self._spark.sparkContext.setJobGroup(name, name)
        self._timer_base = time.time()
        print(desc)

    def _clear_job_group(self) -> None:
        # TODO: Uses `SparkContext.clearJobGroup()` instead
        self._spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        self._spark.sparkContext.setLocalProperty("spark.job.description", None)
        self._spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)

    def _end_spark_jobs(self) -> None:
        print(f"Elapsed time is {time.time() - self._timer_base}(s)")  # type: ignore
        self._clear_job_group()

    def _release_resources(self, env: Dict[str, str]) -> None:
        for t in list(filter(lambda x: x in self._meta_view_names, env.values())):
            self._spark.sql(f"DROP VIEW IF EXISTS {env[t]}")

    def _check_input(self) -> Tuple[Dict[str, str], str, List[str]]:
        env = json.loads(self._svg_api.checkInputTable(self.db_name, self.table_name, self.row_id))
        continous_attrs = env["continous_attrs"].split(",")
        return env, self._spark.table(env["input_table"]), \
            continous_attrs if continous_attrs != [""] else []

    def _detect_error_cells(self, input_table: str) -> str:
        # Initializes the given error detectors with the input params
        for d in self.error_detectors:
            d.setUp(self.row_id, input_table)  # type: ignore

        error_cells_dfs = [d.detect() for d in self.error_detectors]

        err_cells = self._temp_name("gray_cells")
        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        err_cells_df.distinct().cache().createOrReplaceTempView(err_cells)
        return err_cells

    def _detect_errors(self, env: Dict[str, str]) -> str:
        # If `self.error_cells` provided, just uses it
        if self.error_cells is not None:
            df = self.error_cells if isinstance(self.error_cells, DataFrame) \
                else self._spark.table(self.error_cells)
            if not all(c in df.columns for c in (self.row_id, "attribute")):
                raise ValueError(f"Error cells must have `{self.row_id}` and "
                                 "`attribute` in columns")

            env["gray_cells"] = self._temp_view(df) if isinstance(self.error_cells, DataFrame) \
                else str(self.error_cells)
            print('[Error Detection Phase] Error cells provided by `{env["gray_cells"]}`')

            # We assume that the given error cells are true, so we skip computing error domains
            # with probability because the computational cost is much high.
            self.domain_threshold_beta = 1.0
        else:
            # Applys error detectors to get gray cells
            self._start_spark_jobs(
                "error detection",
                f'[Error Detection Phase] Detecting errors in a table `{env["input_table"]}` '
                f'({env["num_attrs"]} cols x {env["num_input_rows"]} rows)...')
            env["gray_cells"] = self._detect_error_cells(env["input_table"])
            self._end_spark_jobs()

        return self._spark.table(env["gray_cells"])

    def _prepare_repair_base(self, env: Dict[str, str], gray_cells_df: DataFrame) -> str:
        # Sets NULL at the detected gray cells
        logging.info("{}/{} suspicious cells found, then converts them into NULL cells...".format(
            gray_cells_df.count(), int(env["num_input_rows"]) * int(env["num_attrs"])))
        env.update(json.loads(self._svg_api.convertErrorCellsToNull(
            env["input_table"], env["gray_cells"],
            self.row_id)))

        return self._spark.table(env["repair_base"])

    def _preprocess(self, env: Dict[str, str], continous_attrs: List[str]) -> DataFrame:
        # Filters out attributes having large domains and makes continous values
        # discrete if necessary.
        env.update(json.loads(self._svg_api.convertToDiscreteFeatures(
            self.db_name, self.table_name, self.row_id,
            self.discrete_thres)))

        discrete_ft_df = self._spark.table(env["discrete_features"])
        logging.info("Valid {} attributes ({}) found in the {} input attributes ({}) and "
                     "{} continous attributes ({}) included in them".format(
                         len(discrete_ft_df.columns),
                         ",".join(discrete_ft_df.columns),
                         len(self._spark.table(env["input_table"]).columns),
                         ",".join(self._spark.table(env["input_table"]).columns),
                         len(continous_attrs),
                         ",".join(continous_attrs)))

        return discrete_ft_df

    def _analyze_error_cell_domain(self, env: Dict[str, str], gray_cells_df: DataFrame,
                                   continous_attrs: List[str]) -> str:
        # Checks if attributes are discrete or not, and discretizes continous ones
        discrete_ft_df = self._preprocess(env, continous_attrs)

        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        logging.info("Collecting and sampling attribute stats (ratio={} threshold={}) "
                     "before computing error domains...".format(
                         self.attr_stat_sample_ratio,
                         self.attr_stat_threshold))
        env.update(json.loads(self._svg_api.computeAttrStats(
            env["discrete_features"], env["gray_cells"], self.row_id,
            self.attr_stat_sample_ratio,
            self.attr_stat_threshold)))

        self._start_spark_jobs(
            "cell domains analysis",
            "[Error Detection Phase] Analyzing cell domains to fix error cells...")
        env.update(json.loads(self._svg_api.computeDomainInErrorCells(
            env["discrete_features"], env["attr_stats"], env["gray_cells"], self.row_id,
            env["continous_attrs"],
            self.max_attrs_to_compute_domains,
            self.min_corr_thres,
            self.domain_threshold_alpha,
            self.domain_threshold_beta)))
        self._end_spark_jobs()

        return self._spark.table(env["cell_domain"])

    def _extract_error_cells(self, env: Dict[str, str], cell_domain_df: DataFrame,
                             repair_base_df: DataFrame) -> DataFrame:
        # Fixes cells if an inferred value is the same with an initial one
        fix_cells_expr = "if(current_value = domain[0].n, current_value, NULL) value"
        weak_df = cell_domain_df.selectExpr(
            self.row_id, "attribute", "current_value", fix_cells_expr).cache()
        error_cells_df = weak_df.where("value IS NULL").drop("value").cache()
        env["weak"] = self._temp_name("weak")
        weak_df = weak_df.where("value IS NOT NULL") \
            .selectExpr(self.row_id, "attribute", "value repaired")
        weak_df.cache().createOrReplaceTempView(env["weak"])
        ret_as_json = self._svg_api.repairAttrsFrom(
            env["weak"], "", env["repair_base"], self.row_id)
        env["partial_repaired"] = json.loads(ret_as_json)["repaired"]

        print('[Error Detection Phase] {} suspicious cells fixed and '
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
        temp_view = self._temp_name()
        df.createOrReplaceTempView(temp_view)
        ret_as_json = self._svg_api.convertToHistogram(temp_view, self.discrete_thres)
        return self._spark.table(json.loads(ret_as_json)["histogram"])

    def _show_histogram(self, df: DataFrame) -> None:
        import matplotlib.pyplot as plt
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
        print("[Repair Model Training Phase] Sampling {} training data (ratio={}) "
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
            logging.info(f"{y}: #errors={error_num_map[y]}")

        return target_columns

    def _domain_size_based_order(self, env: Dict[str, str], train_df: DataFrame,
                                 error_attrs: List[Row]) -> List[str]:
        # Computes domain sizes for training data
        self._start_spark_jobs(
            "training data stat analysis",
            "[Repair Model Training Phase] Collecting training data stats before "
            "building ML models...")
        env["train"] = self._temp_name("train")
        train_df.createOrReplaceTempView(env["train"])
        env.update(json.loads(self._svg_api.computeDomainSizes(env["train"])))
        self._end_spark_jobs()

        # Sorts target columns by domain size
        target_columns = list(map(lambda row: row.attribute,
                              sorted(error_attrs,
                                     key=lambda row: int(env["distinct_stats"][row.attribute]),
                                     reverse=False)))
        for y in target_columns:
            logging.info(f'{y}: |domain|={env["distinct_stats"][y]}')

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
            logging.info("corr={}, y({})<=X({})".format(-t[0], t[1], ",".join(features)))
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
        excluded_columns.remove(y)

        # Selects features if necessary
        if self.max_training_column_num is not None and \
                int(self.max_training_column_num) < len(features):
            heap: List[Tuple[float, str]] = []
            for f, corr in map(lambda x: tuple(x), env["pairwise_attr_stats"][y]):
                if f in features:
                    # Converts to a negative value for extracting higher values
                    heapq.heappush(heap, (-float(corr), f))

            fts = [heapq.heappop(heap)[1] for i in range(int(self.max_training_column_num))]
            logging.info("Select {} relevant features ({}) from available ones ({})".format(
                len(fts), ",".join(fts), ",".join(features)))
            features = fts

        return features

    def _transform_features(self, env: Dict[str, str], X: pd.DataFrame, features: List[str],
                            continous_attrs: List[str]) -> Tuple[pd.DataFrame, Any]:
        # Transforms discrete attributes with some categorical encoders if necessary
        import category_encoders as ce
        discrete_columns = [c for c in features if c not in continous_attrs]
        if len(discrete_columns) == 0:
            # TODO: Needs to normalize continous values
            transformers = None
        else:
            transformers = []
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
            logging.info("{} encoders transform ({})=>({})".format(
                len(transformers), ",".join(features), ",".join(X.columns)))

        return X, transformers

    def _build_model(self, X: pd.DataFrame, y: pd.DataFrame, is_discrete: bool) -> pd.DataFrame:
        import lightgbm as lgb
        if is_discrete:
            clf = lgb.LGBMClassifier(
                boosting_type="gbdt",
                # objective="multiclass",
                learning_rate=0.1,
                n_estimators=100,
                num_leaves=self.lgb_num_leaves,
                max_depth=self.lgb_max_depth,
                class_weight="balanced"
            )
            return clf.fit(X, y)
        else:
            reg = lgb.LGBMRegressor(
                boosting_type="gbdt",
                objective="regression",
                learning_rate=0.1,
                n_estimators=100,
                num_leaves=self.lgb_num_leaves,
                max_depth=self.lgb_max_depth,
                class_weight="balanced"
            )
            return reg.fit(X, y)

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
        self._start_spark_jobs(
            "repair model training",
            f"[Repair Model Training Phase] Building {len(target_columns)} ML models "
            "to repair the error cells...")
        models = {}
        train_pdf = train_df.toPandas()
        excluded_columns = copy.deepcopy(target_columns)
        for index, y in enumerate(target_columns):
            features = self._select_features(env, train_pdf.columns, y, excluded_columns)
            X, transformers = self._transform_features(
                env, train_pdf[features], features, continous_attrs)
            is_discrete = y not in continous_attrs
            models[y] = (self._build_model(X, train_pdf[y], is_discrete), features, transformers)
            logging.info("{}[{}]: #features={}, y({})<=X({})".format(
                "Classifier" if is_discrete else "Regressor", index, len(X.columns), y,
                ",".join(features)))
        self._end_spark_jobs()

        return models, target_columns

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
        grouping_key = self._temp_name("__grouping_key")
        env["dirty"] = self._temp_name("dirty")
        dirty_df.createOrReplaceTempView(env["dirty"])
        dirty_df = dirty_df.withColumn(
            grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))

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
                    if transformers is not None:
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
        self._start_spark_jobs(
            "repairing",
            f"[Repairing Phase] Computing {error_cells_df.count()} repair updates in "
            f"{dirty_df.count()} rows...")
        repaired_df = dirty_df.groupBy(grouping_key).apply(repair).drop(grouping_key).cache()
        repaired_df.write.format("noop").mode("overwrite").save()
        self._end_spark_jobs()
        return repaired_df

    def _compute_repair_pmf(self, repaired_df: DataFrame, error_cells_df: DataFrame) -> DataFrame:
        parse_pmf_json_expr = "from_json(value, 'classes array<string>, probs array<double>') pmf"
        to_pmf_expr = "arrays_zip(pmf.classes, pmf.probs) pmf"
        to_current_expr = "named_struct('value', current_value, 'prob', " \
            "coalesce(pmf.probs[array_position(pmf.classes, current_value) - 1], 0.0)) current"
        sorted_pmf_expr = "array_sort(pmf, (left, right) -> if(left.`1` < right.`1`, 1, -1)) pmf"
        pmf_df = self._flatten(repaired_df) \
            .join(error_cells_df, [self.row_id, "attribute"], "inner") \
            .selectExpr(self.row_id, "attribute", "current_value", parse_pmf_json_expr) \
            .selectExpr(self.row_id, "attribute", to_current_expr, to_pmf_expr) \
            .selectExpr(self.row_id, "attribute", "current", sorted_pmf_expr)

        return pmf_df

    def _maximal_likelihood_repair(self, env: Dict[str, str], repaired_df: DataFrame,
                                   error_cells_df: DataFrame) -> DataFrame:
        # A “Maximal Likelihood Repair” problem defined in the SCARE [2] paper is as follows;
        # Given a scalar \delta and a database D = D_{e} \cup D_{c}, the problem is to
        # find another database instance D' = D'_{e} \cup D_{c} such that L(D'_{e} \| D_{c})
        # is maximum subject to the constraint Dist(D, D') <= \delta.
        # L is a likelihood function and Dist is an arbitrary distance function
        # (e.g., edit distances) between the two database instances D and D'.
        pmf_df = self._compute_repair_pmf(repaired_df, error_cells_df)

        broadcasted_distance = self._spark.sparkContext.broadcast(self.distance)

        @functions.pandas_udf("double", functions.PandasUDFType.SCALAR)
        def distance(xs: pd.Series, ys: pd.Series) -> pd.Series:
            distance = broadcasted_distance.value
            dists = [distance.compute(x, y) for x, y in zip(xs, ys)]
            return pd.Series(dists)

        maximal_likelihood_repair_expr = "named_struct('value', pmf[0].`0`, " \
            "'prob', pmf[0].`1`) repaired"
        score_expr = "ln(repaired.prob / IF(current.prob > 0.0, current.prob, 1e-6)) " \
            "* (1.0 / (1.0 + distance)) score"
        score_df = pmf_df \
            .selectExpr(self.row_id, "attribute", "current", maximal_likelihood_repair_expr) \
            .withColumn("distance", distance(col("current.value"), col("repaired.value"))) \
            .selectExpr(self.row_id, "attribute", "current.value current_value",
                        "repaired.value repaired", score_expr)

        assert self.repair_delta is not None
        num_error_cells = error_cells_df.count()
        percent = min(1.0, self.repair_delta / num_error_cells)
        percentile = score_df.selectExpr(f"percentile(score, {percent}) thres").collect()[0]
        top_delta_repairs_expr = \
            f"IF(score <= {percentile.thres}, repaired, current_value) repaired"
        top_delta_repairs_df = score_df.selectExpr(self.row_id, "attribute", top_delta_repairs_expr)
        print("[Repairing Phase] {} repair updates (delta={}) selected "
              "among {} candidates...".format(
                  score_df.where(f"score <= {percentile.thres}").count(),
                  self.repair_delta,
                  num_error_cells))

        return top_delta_repairs_df

    def _run(self, env: Dict[str, str], input_df: DataFrame, continous_attrs: List[str],
             detect_errors_only: bool, compute_training_target_hist: bool,
             compute_repair_candidate_prob: bool, repair_data: bool) -> DataFrame:
        #################################################################################
        # 1. Error Detection Phase
        #################################################################################

        # If no error found, it just returns the given table
        gray_cells_df = self._detect_errors(env)
        if gray_cells_df.count() == 0:  # type: ignore
            print("Any error cells not found, so returns the input as clean cells")
            self._release_resources(env)
            return input_df

        # Sets NULL to suspicious cells
        repair_base_df = self._prepare_repair_base(env, gray_cells_df)

        # Selects error cells based on the result of domain analysis
        cell_domain_df = self._analyze_error_cell_domain(env, gray_cells_df, continous_attrs)

        # If no error cell found, ready to return a clean table
        error_cells_df = self._extract_error_cells(env, cell_domain_df, repair_base_df)
        if error_cells_df.count() == 0:
            self._release_resources(env)
            return input_df

        # If `detect_errors_only` is True, returns found error cells
        if detect_errors_only:
            self._release_resources(env)
            return error_cells_df

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
            return self._compute_repair_pmf(repaired_df, error_cells_df)

        # If any discrete target columns and its probability distribution given,
        # computes scores to decide which cells should be repaired to follow the
        # “Maximal Likelihood Repair” problem.
        if self.maximal_likelihood_repair_enabled:
            top_delta_repairs_df = self._maximal_likelihood_repair(env, repaired_df, error_cells_df)
            if not repair_data:
                return top_delta_repairs_df

            # If `repair_data` is True, applys the selected repair updates into `dirty`
            env["top_delta_repairs"] = self._temp_name("top_delta_repairs")
            top_delta_repairs_df.createOrReplaceTempView(env["top_delta_repairs"])
            env.update(json.loads(self._svg_api.repairAttrsFrom(env["top_delta_repairs"],
                                  "", env["dirty"], self.row_id)))
            repaired_df = self._spark.table(env["repaired"])

        # If `repair_data` is False, returns repair candidates whoes
        # value is the same with `current_value`.
        if not repair_data:
            repair_candidates_df = self._flatten(repaired_df) \
                .join(error_cells_df, [self.row_id, "attribute"], "inner") \
                .selectExpr("tid", "attribute", "current_value", "value repaired") \
                .where("not(current_value <=> repaired)")
            return repair_candidates_df
        else:
            clean_df = fixed_df.union(repaired_df)
            assert clean_df.count() == input_df.count()
            return clean_df

    def run(self, detect_errors_only: bool = False, compute_repair_candidate_prob: bool = False,
            compute_training_target_hist: bool = False, repair_data: bool = False) -> DataFrame:
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before repairing")
        if self.maximal_likelihood_repair_enabled and self.repair_delta is None:
            raise ValueError("`setRepairDelta` should be called before "
                             "maximal likelihood repairing")
        if self.inference_order not in ["error", "domain", "entropy"]:
            raise ValueError(f"Inference order must be `error`, `domain`, or `entropy`, "
                             "but `{self.inference_order}` found")

        exclusive_param_list = [
            detect_errors_only, compute_repair_candidate_prob,
            compute_training_target_hist, repair_data]
        if exclusive_param_list.count(True) > 1:
            raise ValueError("`detect_errors_only`, `compute_repair_candidate_prob`, "
                             "`compute_training_target_hist`, and `repair_data` cannot "
                             "be set to True simultaneously")

        # If `self.repair_updates` specified, just applies repair updates
        if self.repair_updates is not None:
            repair_updates_view = self._temp_view(self.repair_updates) \
                if isinstance(self.repair_updates, DataFrame) else str(self.repair_updates)
            ret_as_json = self._svg_api.repairAttrsFrom(
                repair_updates_view, self.db_name, self.table_name, self.row_id)
            return self._spark.table(json.loads(ret_as_json)["repaired"])

        # Checks # of input rows and attributes
        env, input_df, continous_attrs = self._check_input()

        if compute_repair_candidate_prob and len(continous_attrs) != 0:
            raise ValueError("Cannot compute probability mass function of repairs "
                             "when continous attributes found")
        if self.maximal_likelihood_repair_enabled and len(continous_attrs) != 0:
            raise ValueError("Cannot enable maximal likelihood repair mode "
                             "when continous attributes found")

        __start = time.time()
        df = self._run(env, input_df, continous_attrs, detect_errors_only,
                       compute_training_target_hist, compute_repair_candidate_prob, repair_data)
        print(f"!!!Total processing time is {time.time() - __start}(s)!!!")
        return df

    @staticmethod
    def version() -> str:
        # TODO: Extracts a version string from the root pom.xml
        return "0.1.0-spark3.0-EXPERIMENTAL"
