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

from pyspark.sql import DataFrame, functions
from pyspark.sql.functions import col

from repair.base import *
from repair.detectors import *
from repair.distances import *


class ScavengerRepairModel(ApiBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output="", db_name=""):
        super().__init__()

        # Basic parameters
        self.constraint_input_path = None
        self.db_name = db_name
        self.table_name = None
        self.row_id = None

        # Parameters for error detection
        self.error_cells = None
        self.discrete_thres = 80
        self.min_corr_thres = 0.70
        self.domain_threshold_alpha = 0.0
        self.domain_threshold_beta = 0.70
        self.max_attrs_to_compute_domains = 4
        self.attr_stat_sample_ratio = 1.0
        self.attr_stat_threshold = 0.0

        # Parameters for repair model training
        self.training_data_sample_ratio = 1.0
        self.max_training_column_num = None
        self.small_domain_threshold = 12
        self.inference_order = "entropy"
        self.lgb_num_leaves = 31
        self.lgb_max_depth = -1

        # Parameters for repairing
        self.repair_updates = None
        self.maximal_likelihood_repair_enabled = False
        self.repair_delta = None

        # JVM interfaces for Scavenger APIs
        self.__svg_api = self.jvm.ScavengerRepairApi

        # Internally used to check elapsed time
        self.__timer_base = None

        # Defines detectors to discover error cells
        # TODO: Needs to support a pattern-based error detector (See [10])
        self.__detectors = [
            NullErrorDetector(),
            ConstraintErrorDetector(),
            # OutlierErrorDetector()
        ]

        # Defines a class to compute cost of updates.
        #
        # TODO: Needs a sophisticated way to compute distances between a current value and a repair candidate.
        # For example, the HoloDetect paper [1] proposes a noisy channel model for the data augmentation methodology
        # of training data. This model consists of transformation rules and and data augmentation policies
        # (i.e., distribution over those data transformation). This model can be re-used to compute this cost.
        # For more details, see the section 5, 'DATA AUGMENTATION LEARNING', in the paper.
        self.__distance = Levenshtein()

        # Temporary views used in repairing processes
        self.__meta_view_names = [
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

    def setConstraints(self, constraint_input_path):
        self.constraint_input_path = constraint_input_path
        return self

    def setDbName(self, db_name):
        self.db_name = db_name
        return self

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setErrorCells(self, error_cells):
        self.error_cells = error_cells
        return self

    def setDiscreteThreshold(self, thres):
        self.discrete_thres = thres
        return self

    def setMinCorrThreshold(self, thres):
        self.min_corr_thres = thres
        return self

    def setDomainThresholds(self, alpha, beta):
        self.domain_threshold_alpha = alpha
        self.domain_threshold_beta = beta
        return self

    def setAttrMaxNumToComputeDomains(self, max):
        self.max_attrs_to_compute_domains = max
        return self

    def setAttrStatSampleRatio(self, ratio):
        self.attr_stat_sample_ratio = ratio
        return self

    def setAttrStatThreshold(self, ratio):
        self.attr_stat_threshold = ratio
        return self

    def setTrainingDataSampleRatio(self, ratio):
        self.training_data_sample_ratio = ratio
        return self

    def setMaxTrainingColumnNum(self, n):
        self.max_training_column_num = n
        return self

    def setSmallDomainThreshold(self, thres):
        self.small_domain_threshold = thres
        return self

    def setInferenceOrder(self, inference_order):
        self.inference_order = inference_order
        return self

    def setLGBNumLeaves(self, n):
        self.lgb_num_leaves = n
        return self

    def setLGBMaxDepth(self, n):
        self.lgb_max_depth = n
        return self

    def setRepairUpdates(self, repair_updates):
        self.repair_updates = repair_updates
        return self

    def setMaximalLikelihoodRepairEnabled(self, enabled):
        self.maximal_likelihood_repair_enabled = enabled
        return self

    def setRepairDelta(self, delta):
        if int(delta) <= 0:
            raise ValueError("repair delta must be positive")
        self.repair_delta = delta
        return self

    def __temp_name(self, prefix="temp"):
        return "%s_%s" % (prefix, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    def __temp_view(self, df):
        temp_view = self.__temp_name()
        df.createOrReplaceTempView(temp_view)
        return temp_view

    def __flatten(self, df):
        temp_view = self.__temp_name()
        df.createOrReplaceTempView(temp_view)
        jdf = self.jvm.ScavengerMiscApi.flattenTable("", temp_view, self.row_id)
        return DataFrame(jdf, self.spark._wrapped)

    def __start_spark_jobs(self, name, desc):
        self.spark.sparkContext.setJobGroup(name, name)
        self.__timer_base = time.time()
        self.outputToConsole(desc)

    def __clear_job_group(self):
        # TODO: Uses `SparkContext.clearJobGroup()` instead
        self.spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        self.spark.sparkContext.setLocalProperty("spark.job.description", None)
        self.spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)

    def __end_spark_jobs(self):
        self.outputToConsole("Elapsed time is %s(s)" % (time.time() - self.__timer_base))
        self.__clear_job_group()

    def __release_resources(self, env):
        for t in list(filter(lambda x: x in self.__meta_view_names, env.values())):
            self.spark.sql("DROP VIEW IF EXISTS %s" % env[t])

    def __check_input(self, env):
        env.update(json.loads(self.__svg_api.checkInputTable(self.db_name, self.table_name, self.row_id)))
        continous_attrs = env["continous_attrs"].split(",")
        return self.spark.table(env["input_table"]), \
            continous_attrs if continous_attrs != [""] else []

    def __detect_error_cells(self, env):
        # Initializes defined error detectors with the given env
        for d in self.__detectors:
            d.setup(env)

        error_cells_dfs = [ d.detect() for d in self.__detectors ]

        err_cells = self.__temp_name("gray_cells")
        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        err_cells_df.cache().createOrReplaceTempView(err_cells)
        return err_cells

    def __detect_errors(self, env):
        # If `self.error_cells` provided, just uses it
        if self.error_cells is not None:
            df = self.error_cells if isinstance(self.error_cells, DataFrame) \
                else self.spark.table(self.error_cells)
            if not all(c in df.columns for c in (self.row_id, "attribute")):
                raise ValueError("error cells must have `%s` and `attribute` in columns" % self.row_id)

            env["gray_cells"] = self.__temp_view(df) if isinstance(self.error_cells, DataFrame) \
                else str(self.error_cells)
            self.outputToConsole("[Error Detection Phase] Error cells provided by `%s`" % \
                str(env["gray_cells"]))

            # We assume that the given error cells are true, so we skip computing error domains
            # with probability because the computational cost is much high.
            self.domain_threshold_beta = 1.0
        else:
            # Applys error detectors to get gray cells
            self.__start_spark_jobs("error detection",
                "[Error Detection Phase] Detecting errors in a table `%s` (%s cols x %s rows)..." % \
                    (env["input_table"], env["num_attrs"], env["num_input_rows"]))
            env["gray_cells"] = self.__detect_error_cells(env)
            self.__end_spark_jobs()

        return self.spark.table(env["gray_cells"])

    def __prepare_repair_base(self, env, gray_cells_df):
        # Sets NULL at the detected gray cells
        logging.info("%s/%s suspicious cells found, then converts them into NULL cells..." % \
            (gray_cells_df.count(), int(env["num_input_rows"]) * int(env["num_attrs"])))
        env.update(json.loads(self.__svg_api.convertErrorCellsToNull(
            env["input_table"], env["gray_cells"],
            self.row_id)))

        return self.spark.table(env["repair_base"])

    def __preprocess(self, env, continous_attrs):
        # Filters out attributes having large domains and makes continous values
        # discrete if necessary.
        env.update(json.loads(self.__svg_api.convertToDiscreteFeatures(
            self.db_name, self.table_name, self.row_id,
            self.discrete_thres)))

        discrete_ft_df = self.spark.table(env["discrete_features"])
        logging.info("Valid %s attributes (%s) found in the %s input attributes (%s) and " \
            "%s continous attributes (%s) included in them" % ( \
                len(discrete_ft_df.columns), \
                ",".join(discrete_ft_df.columns), \
                len(self.spark.table(env["input_table"]).columns), \
                ",".join(self.spark.table(env["input_table"]).columns), \
                len(continous_attrs), \
                ",".join(continous_attrs)))

        return discrete_ft_df

    def __analyze_error_cell_domain(self, env, gray_cells_df, continous_attrs):
        # Checks if attributes are discrete or not, and discretizes continous ones
        discrete_ft_df = self.__preprocess(env, continous_attrs)

        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        logging.info("Collecting and sampling attribute stats (ratio=%s threshold=%s) " \
                "before computing error domains..." % \
                (self.attr_stat_sample_ratio, self.attr_stat_threshold))
        env.update(json.loads(self.__svg_api.computeAttrStats(
            env["discrete_features"], env["gray_cells"], self.row_id,
            self.attr_stat_sample_ratio,
            self.attr_stat_threshold)))

        self.__start_spark_jobs("cell domains analysis",
            "[Error Detection Phase] Analyzing cell domains to fix error cells...")
        env.update(json.loads(self.__svg_api.computeDomainInErrorCells(
            env["discrete_features"], env["attr_stats"], env["gray_cells"], env["row_id"],
            env["continous_attrs"],
            self.max_attrs_to_compute_domains,
            self.min_corr_thres,
            self.domain_threshold_alpha,
            self.domain_threshold_beta)))
        self.__end_spark_jobs()

        return self.spark.table(env["cell_domain"])

    def __extract_error_cells(self, env, cell_domain_df, repair_base_df):
        # Fixes cells if an inferred value is the same with an initial one
        fix_cells_expr = "if(current_value = domain[0].n, current_value, NULL) value"
        weak_df = cell_domain_df.selectExpr(self.row_id, "attribute", "current_value", fix_cells_expr).cache()
        error_cells_df = weak_df.where("value IS NULL").drop("value").cache()
        env["weak"] = self.__temp_name("weak")
        weak_df = weak_df.where("value IS NOT NULL").selectExpr(self.row_id, "attribute", "value repaired")
        weak_df.cache().createOrReplaceTempView(env["weak"])
        ret_as_json = self.__svg_api.repairAttrsFrom(env["weak"], "", env["repair_base"], self.row_id)
        env["partial_repaired"] = json.loads(ret_as_json)["repaired"]

        self.outputToConsole("[Error Detection Phase] %d suspicious cells fixed and %d error cells remaining..." % \
            (self.spark.table(env["weak"]).count(), error_cells_df.count()))

        return error_cells_df

    def __split_clean_and_dirty_rows(self, env, error_cells_df):
        error_rows_df = error_cells_df.selectExpr(self.row_id).distinct().cache()
        fixed_df = self.spark.table(env["partial_repaired"]).join(error_rows_df, self.row_id, "left_anti").cache()
        dirty_df = self.spark.table(env["partial_repaired"]).join(error_rows_df, self.row_id, "left_semi").cache()
        error_attrs = error_cells_df.groupBy("attribute").agg(functions.count("attribute").alias("cnt")).collect()
        assert len(error_attrs) > 0
        return fixed_df, dirty_df, error_attrs

    def __convert_to_histogram(self, df):
        temp_view = self.__temp_name()
        df.createOrReplaceTempView(temp_view)
        ret_as_json = self.__svg_api.convertToHistogram(temp_view, self.discrete_thres)
        return self.spark.table(json.loads(ret_as_json)["histogram"])

    def __show_histogram(self, df):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        num_targets = df.count()
        for index, row in enumerate(df.collect()):
            pdf = df.where("attribute = '%s'" % row.attribute).selectExpr("inline(histogram)").toPandas()
            f = fig.add_subplot(num_targets, 1, index + 1)
            f.bar(pdf["value"], pdf["cnt"])
            f.set_xlabel(row.attribute)
            f.set_ylabel("cnt")

        fig.tight_layout()
        fig.show()

    def __select_training_rows(self, fixed_df):
        # Prepares training data to repair the remaining error cells
        # TODO: Needs more smart sampling, e.g., down-sampling
        train_df = fixed_df.sample(self.training_data_sample_ratio).drop(self.row_id).cache()
        self.outputToConsole("[Repair Model Training Phase] Sampling %s training data (ratio=%s) from %s clean rows..." % \
            (train_df.count(), self.training_data_sample_ratio, fixed_df.count()))
        return train_df

    def __error_num_based_order(self, error_attrs):
        # Sorts target columns by the number of errors
        error_num_map = {}
        for row in error_attrs:
            error_num_map[row.attribute] = row.cnt

        target_columns = list(map(lambda row: row.attribute, \
            sorted(error_attrs, key=lambda row: row.cnt, reverse=False)))
        for y in target_columns:
            logging.info("%s: #errors=%s" % (y, error_num_map[y]))

        return target_columns

    def __domain_size_based_order(self, env, train_df, error_attrs):
        # Computes domain sizes for training data
        self.__start_spark_jobs("training data stat analysis",
            "[Repair Model Training Phase] Collecting training data stats before building ML models...")
        env["train"] = self.__temp_name("train")
        train_df.createOrReplaceTempView(env["train"])
        env.update(json.loads(self.__svg_api.computeDomainSizes(env["train"])))
        self.__end_spark_jobs()

        # Sorts target columns by domain size
        target_columns = list(map(lambda row: row.attribute, \
            sorted(error_attrs, key=lambda row: int(env["distinct_stats"][row.attribute]), reverse=False)))
        for y in target_columns:
            logging.info("%s: |domain|=%s" % (y, env["distinct_stats"][y]))

        return target_columns

    def __entropy_based_order(self, env, train_df, error_attrs):
        # Sorts target columns by correlations
        target_columns = []
        error_attrs = list(map(lambda row: row.attribute, error_attrs))

        for index in range(len(error_attrs)):
            features = [ c for c in train_df.columns if c not in error_attrs ]
            targets = []
            for c in error_attrs:
                total_corr = 0.0
                for f, corr in map(lambda x: tuple(x), env["pairwise_attr_stats"][c]):
                    if f in features:
                        total_corr += float(corr)

                heapq.heappush(targets, (-total_corr, c))

            t = heapq.heappop(targets)
            target_columns.append(t[1])
            logging.info("corr=%s, y(%s)<=X(%s)" % (-t[0], t[1], ",".join(features)))
            error_attrs.remove(t[1])

        return target_columns

    def __compute_inference_order(self, env, train_df, error_attrs):
        # Defines a inference order based on `train_df`.
        #
        # TODO: Needs to analyze more dependencies (e.g., based on graph algorithms) between
        # target columns and the other ones for decideing a inference order.
        # For example, the SCARE paper [2] builds a dependency graph (a variant of graphical models)
        # to analyze the correlatioin of input data. But, the analysis is compute-intensive, so
        # we just use a naive approache to define the order now.
        if self.inference_order == "entropy":
            return self.__entropy_based_order(env, train_df, error_attrs)
        elif self.inference_order == "domain":
            return self.__domain_size_based_order(env, train_df, error_attrs)
        elif self.inference_order == "error":
            return self.__error_num_based_order(error_attrs)

    def __select_features(self, env, input_columns, y, excluded_columns):
        # All the available features
        features = [c for c in input_columns if c not in excluded_columns]
        excluded_columns.remove(y)

        # Selects features if necessary
        if self.max_training_column_num is not None and \
                int(self.max_training_column_num) < len(features):
            fts = []
            for f, corr in map(lambda x: tuple(x), env["pairwise_attr_stats"][y]):
                if f in features:
                   # Converts to a negative value for extracting higher values
                   heapq.heappush(fts, (-float(corr), f))

            fts = [ heapq.heappop(fts)[1] for i in range(int(self.max_training_column_num)) ]
            logging.info("Select %s relevant features (%s) from available ones (%s)" % \
                (len(fts), ",".join(fts), ",".join(features)))
            features = fts

        return features

    def __transform_features(self, env, X, features, continous_attrs):
        # Transforms discrete attributes with some categorical encoders if necessary
        import category_encoders as ce
        discrete_columns = [ c for c in features if c not in continous_attrs ]
        if len(discrete_columns) == 0:
            # TODO: Needs to normalize continous values
            transformers = None
        else:
            transformers = []
            # TODO: Needs to reconsider feature transformation in this part, e.g.,
            # we can use `ce.OrdinalEncoder` for small domain features.
            # For the other category encoders, see https://github.com/scikit-learn-contrib/category_encoders
            small_domain_columns = [ c for c in discrete_columns \
                if int(env["distinct_stats"][c]) < self.small_domain_threshold ]
            discrete_columns = [ c for c in discrete_columns \
                if c not in small_domain_columns ]
            if len(small_domain_columns) > 0:
                transformers.append(ce.SumEncoder(cols=small_domain_columns, handle_unknown='impute'))
            if len(discrete_columns) > 0:
                transformers.append(ce.OrdinalEncoder(cols=discrete_columns, handle_unknown='impute'))
            # TODO: Needs to include `dirty_df` in this transformation
            for transformer in transformers:
                X = transformer.fit_transform(X)
            logging.info("%s encoders transform (%s)=>(%s)" % \
                (len(transformers), ",".join(features), ",".join(X.columns)))

        return X, transformers

    def __build_model(self, X, y, is_discrete):
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

    def __build_repair_models(self, env, train_df, error_attrs, continous_attrs):
        # We now employ a simple repair model based on the SCARE paper [2] for scalable processing
        # on Apache Spark. Given a database tuple t = ce (c: correct attribute values, e: error attribute values),
        # the conditional probability of each combination of the error attribute values c can be
        # computed the product rule:
        #
        #  P(e\|c)=P(e[E_{1}]\|c)\prod_{i=2}^{K}P(e[E_{i}]\|c,e[E_{1}...E_{i-1}])
        #
        # where K is the number of error attributes, `len(error_attrs)`, and {E_[1], ..., E_[K]} is
        # a particular dependency order in error attributes. More sophisticated repair models have been
        # proposed recently, e.g., a Markov logic network based model in HoloClean [4].
        # Therefore, we might be able to improve our model more baesd on
        # the-state-of-the-art approaches.

        # Computes a inference order based on dependencies between `error_attrs` and the others
        target_columns = self.__compute_inference_order(env, train_df, error_attrs)

        # Builds multiple ML models to repair error cells
        self.__start_spark_jobs("repair model training",
            "[Repair Model Training Phase] Building %s ML models to repair the error cells..." % len(target_columns))
        models = {}
        train_pdf = train_df.toPandas()
        excluded_columns = copy.deepcopy(target_columns)
        for index, y in enumerate(target_columns):
            features = self.__select_features(env, train_pdf.columns, y, excluded_columns)
            X, transformers = self.__transform_features(env, train_pdf[features], features, continous_attrs)
            is_discrete = y not in continous_attrs
            models[y] = (self.__build_model(X, train_pdf[y], is_discrete), features, transformers)
            logging.info("%s[%d]: #features=%s, y(%s)<=X(%s)" % ( \
                "Classifier" if is_discrete else "Regressor", index, len(X.columns), y,
                ",".join(features)))
        self.__end_spark_jobs()

        return models, target_columns

    def __repair(self, env, models, target_columns, continous_attrs, dirty_df, error_cells_df, compute_repair_candidate_prob):
        # Shares all the variables for the learnt models in a Spark cluster
        broadcasted_target_columns = self.spark.sparkContext.broadcast(target_columns)
        broadcasted_continous_attrs = self.spark.sparkContext.broadcast(continous_attrs)
        broadcasted_models = self.spark.sparkContext.broadcast(models)
        broadcasted_compute_repair_candidate_prob = self.spark.sparkContext.broadcast(compute_repair_candidate_prob)
        broadcasted_maximal_likelihood_repair_enabled = \
            self.spark.sparkContext.broadcast(self.maximal_likelihood_repair_enabled)

        # Sets a grouping key for inference
        num_parallelism = self.spark.sparkContext.defaultParallelism
        grouping_key = self.__temp_name("__grouping_key")
        env["dirty"] = self.__temp_name("dirty")
        dirty_df.createOrReplaceTempView(env["dirty"])
        dirty_df = dirty_df.withColumn(grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))

        @functions.pandas_udf(dirty_df.schema, functions.PandasUDFType.GROUPED_MAP)
        def repair(pdf):
            target_columns = broadcasted_target_columns.value
            continous_attrs = broadcasted_continous_attrs.value
            models = broadcasted_models.value
            compute_repair_candidate_prob = broadcasted_compute_repair_candidate_prob.value
            maximal_likelihood_repair_enabled = \
                broadcasted_maximal_likelihood_repair_enabled.value
            rows = []
            for index, row in pdf.iterrows():
                for y in target_columns:
                    (model, features, transformers) = models[y]

                    # Preprocesses the input row for prediction
                    X = pd.DataFrame(row[features]).T
                    for c in [ f for f in features if f in continous_attrs ]:
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
                                pmf = { "classes" : model.classes_.tolist(), "probs" : predicted[0].tolist() }
                                row[y] = json.dumps(pmf)
                            else:
                                predicted = model.predict(X)
                                row[y] = predicted[0]

                rows.append(row)

            return pd.DataFrame(rows)

        # Predicts the remaining error cells based on the trained models.
        # TODO: Might need to compare repair costs (cost of an update, c) to
        # the likelihood benefits of the updates (likelihood benefit of an update, l).
        self.__start_spark_jobs("repairing", "[Repairing Phase] Computing %s repair updates in %s rows..." % \
            (error_cells_df.count(), dirty_df.count()))
        repaired_df = dirty_df.groupBy(grouping_key).apply(repair).drop(grouping_key).cache()
        repaired_df.write.format("noop").mode("overwrite").save()
        self.__end_spark_jobs()
        return repaired_df

    def __compute_repair_pmf(self, repaired_df, error_cells_df):
        parse_pmf_json_expr = "from_json(value, 'classes array<string>, probs array<double>') pmf"
        to_pmf_expr = "arrays_zip(pmf.classes, pmf.probs) pmf"
        to_current_expr = "named_struct('value', current_value, 'prob', " \
            "coalesce(pmf.probs[array_position(pmf.classes, current_value) - 1], 0.0)) current"
        sorted_pmf_expr = "array_sort(pmf, " \
            "(left, right) -> if(left.`1` < right.`1`, 1, -1)) pmf"
        pmf_df = self.__flatten(repaired_df) \
            .join(error_cells_df, [self.row_id, "attribute"], "inner") \
            .selectExpr(self.row_id, "attribute", "current_value", parse_pmf_json_expr) \
            .selectExpr(self.row_id, "attribute", to_current_expr, to_pmf_expr) \
            .selectExpr(self.row_id, "attribute", "current", sorted_pmf_expr)

        return pmf_df

    def __maximal_likelihood_repair(self, env, repaired_df, error_cells_df):
        # A “Maximal Likelihood Repair” problem defined in the SCARE [2] paper is as follows;
        # Given a scalar \delta and a database D = D_{e} \cup D_{c}, the problem is to
        # find another database instance D' = D'_{e} \cup D_{c} such that L(D'_{e} \| D_{c})
        # is maximum subject to the constraint Dist(D, D') <= \delta.
        # L is a likelihood function and Dist is an arbitrary distance function
        # (e.g., edit distances) between the two database instances D and D'.
        pmf_df = self.__compute_repair_pmf(repaired_df, error_cells_df)

        broadcasted_distance = self.spark.sparkContext.broadcast(self.__distance)

        @functions.pandas_udf("double", functions.PandasUDFType.SCALAR)
        def distance(xs, ys):
            distance = broadcasted_distance.value
            dists = [distance.compute(x, y) for x, y in zip(xs, ys)]
            return pd.Series(dists)

        maximal_likelihood_repair_expr = "named_struct('value', pmf[0].`0`, 'prob', pmf[0].`1`) repaired"
        score_expr = "ln(repaired.prob / IF(current.prob > 0.0, current.prob, 1e-6)) * (1.0 / (1.0 + distance)) score"
        score_df = pmf_df \
            .selectExpr(self.row_id, "attribute", "current", maximal_likelihood_repair_expr) \
            .withColumn("distance", distance(col("current.value"), col("repaired.value"))) \
            .selectExpr(self.row_id, "attribute", "current.value current_value", "repaired.value repaired", score_expr)

        assert self.repair_delta is not None
        num_error_cells = error_cells_df.count()
        percent = min(1.0, self.repair_delta / num_error_cells)
        percentile = score_df.selectExpr("percentile(score, %s) thres" % percent).collect()[0]
        top_delta_repairs_expr = "IF(score <= %s, repaired, current_value) repaired" % percentile.thres
        top_delta_repairs_df = score_df.selectExpr(self.row_id, "attribute", top_delta_repairs_expr)
        self.outputToConsole("[Repairing Phase] %s repair updates (delta=%s) selected among %s candidates..." % \
            (score_df.where("score <= %s" % percentile.thres).count(), self.repair_delta, num_error_cells))
        return top_delta_repairs_df

    def __run(self, env, input_df, continous_attrs, detect_errors_only, compute_training_target_hist,
            compute_repair_candidate_prob, repair_data):
        #################################################################################
        # 1. Error Detection Phase
        #################################################################################

        # If no error found, it just returns the given table
        gray_cells_df = self.__detect_errors(env)
        if gray_cells_df.count() == 0:
            self.outputToConsole("Any error cells not found, so returns the input as clean cells")
            self.__release_resources(env)
            return input_df

        # Sets NULL to suspicious cells
        repair_base_df = self.__prepare_repair_base(env, gray_cells_df)

        # Selects error cells based on the result of domain analysis
        cell_domain_df = self.__analyze_error_cell_domain(env, gray_cells_df, continous_attrs)

        # If no error cell found, ready to return a clean table
        error_cells_df = self.__extract_error_cells(env, cell_domain_df, repair_base_df)
        if error_cells_df.count() == 0:
            self.__release_resources(env)
            return input_df

        # If `detect_errors_only` is True, returns found error cells
        if detect_errors_only:
            self.__release_resources(env)
            return error_cells_df

        #################################################################################
        # 2. Repair Model Training Phase
        #################################################################################

        # Selects rows for training, build models, and repair cells
        fixed_df, dirty_df, error_attrs = self.__split_clean_and_dirty_rows(env, error_cells_df)
        if compute_training_target_hist:
            target_columns = list(map(lambda row: row.attribute, error_attrs))
            df = fixed_df.selectExpr(target_columns)
            hist_df = self.__convert_to_histogram(df)
            # self.__show_histogram(hist_df)
            return hist_df

        train_df = self.__select_training_rows(fixed_df)
        if train_df.count() <= float(env["num_input_rows"]) * 0.10:
            raise ValueError("Number of clean training rows must be greater than the 10% number of input rows")

        # Checks if we have the enough number of features for inference
        # TODO: In case of `num_features == 0`, we might be able to select the most accurate and
        # predictable column as a staring feature.
        num_features = len(train_df.columns) - len(error_attrs)
        if num_features == 0:
            raise ValueError("At least %s features needed to repair error cells, " \
                "but %s features found" % num_features)

        models, target_columns = self.__build_repair_models(env, train_df, error_attrs, continous_attrs)

        #################################################################################
        # 3. Repair Phase
        #################################################################################

        repaired_df = self.__repair(env, models, target_columns, continous_attrs,
            dirty_df, error_cells_df, compute_repair_candidate_prob)

        # If `compute_repair_candidate_prob` is True, returns probability mass function of repair candidates
        if compute_repair_candidate_prob:
             return self.__compute_repair_pmf(repaired_df, error_cells_df)

        # If any discrete target columns and its probability distribution given, computes scores
        # to decide which cells should be repaired to follow the “Maximal Likelihood Repair” problem.
        if self.maximal_likelihood_repair_enabled:
            top_delta_repairs_df = self.__maximal_likelihood_repair(env, repaired_df, error_cells_df)
            if not repair_data:
                return top_delta_repairs_df

            # If `repair_data` is True, applys the selected repair updates into `dirty`
            env["top_delta_repairs"] = self.__temp_name("top_delta_repairs")
            top_delta_repairs_df.createOrReplaceTempView(env["top_delta_repairs"])
            env.update(json.loads(self.__svg_api.repairAttrsFrom(env["top_delta_repairs"], "", env["dirty"], self.row_id)))
            repaired_df = self.spark.table(env["repaired"])

        # If `repair_data` is False, returns repair candidates whoes
        # value is the same with `current_value`.
        if not repair_data:
            repair_candidates_df = self.__flatten(repaired_df) \
                .join(error_cells_df, [self.row_id, "attribute"], "inner") \
                .selectExpr("tid", "attribute", "current_value", "value repaired") \
                .where("not(current_value <=> repaired)")
            return repair_candidates_df
        else:
            clean_df = fixed_df.union(repaired_df)
            assert clean_df.count() == input_df.count()
            return clean_df

    def run(self, detect_errors_only=False, compute_repair_candidate_prob=False,
            compute_training_target_hist=False, repair_data=False):
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before repairing")
        if self.maximal_likelihood_repair_enabled and self.repair_delta is None:
            raise ValueError("`setRepairDelta` should be called before maximal likelihood repairing")
        if self.inference_order not in ["error", "domain", "entropy"]:
            raise ValueError("inference order must be `error`, `domain`, or `entropy`, but `%s` found" % \
                self.inference_order)

        exclusive_param_list = [detect_errors_only, compute_repair_candidate_prob, \
            compute_training_target_hist, repair_data]
        if exclusive_param_list.count(True) > 1:
            raise ValueError("`detect_errors_only`, `compute_repair_candidate_prob`, " \
                "`compute_training_target_hist`, and `repair_data` cannot " \
                "be set to True simultaneously")

        # If `self.repair_updates` specified, just applies repair updates
        if self.repair_updates is not None:
            repair_updates_view = self.__temp_view(self.repair_updates) \
                if isinstance(self.repair_updates, DataFrame) else str(self.repair_updates)
            ret_as_json = self.__svg_api.repairAttrsFrom(repair_updates_view, self.db_name, self.table_name, self.row_id)
            return self.spark.table(json.loads(ret_as_json)["repaired"])

        # Env used to repair the given table
        env = {}
        env["row_id"] = self.row_id
        env["constraint_input_path"] = self.constraint_input_path

        # Checks # of input rows and attributes
        input_df, continous_attrs = self.__check_input(env)

        if compute_repair_candidate_prob and len(continous_attrs) != 0:
            raise ValueError("Cannot compute probability mass function of repairs " \
                "when continous attributes found")
        if self.maximal_likelihood_repair_enabled and len(continous_attrs) != 0:
            raise ValueError("Cannot enable maximal likelihood repair mode " \
                "when continous attributes found")

        __start = time.time()
        df = self.__run(env, input_df, continous_attrs, detect_errors_only, compute_training_target_hist,
            compute_repair_candidate_prob, repair_data)
        self.outputToConsole("!!!Total processing time is %s(s)!!!" % (time.time() - __start))
        return df

    def version(self):
        # TODO: Extracts a version string from the root pom.xml
        return "0.1.0-spark3.0-EXPERIMENTAL"

