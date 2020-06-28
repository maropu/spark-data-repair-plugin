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

"""
A Scavenger API Set for Data Profiling & Cleaning
"""

import copy
import datetime
import functools
import json
import logging
import numpy as np
import pandas as pd
import time

from pyspark.sql import DataFrame, SparkSession, functions
from pyspark.sql.functions import col

from repair.detectors import *

class SchemaSpyResult():
    """A result container class for SchemaSpy"""

    # TODO: Prohibit instantiation directly
    def __init__(self, output):
        self.output = output

    def show(self):
        import webbrowser
        webbrowser.open("file://%s/index.html" % self.output)

class SchemaSpyBase():

    def __init__(self):
        self.output = ""
        self.db_name = ""

        self.spark = SparkSession.builder.getOrCreate()

    def setOutput(self, output):
        self.output = output
        return self

    def setDbName(self, db_name):
        self.db_name = db_name
        return self

    def outputToConsole(self, msg):
        print(msg)

class SchemaSpy(SchemaSpyBase):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(SchemaSpy, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self):
        super().__init__()
        self.driver_name = "sqlite"
        self.props = ""
        self.spy_api = sc._jvm.SchemaSpyApi

    @staticmethod
    def getOrCreate():
        return SchemaSpy()

    def setDriverName(self, driver_name):
        self.driver_name = driver_name
        return self

    def setProps(self, props):
        self.props = props
        return self

    def catalogToDataFrame(self):
        jdf = self.spy_api.catalogToDataFrame(self.db_name, self.driver_name, self.props)
        df = DataFrame(jdf, self.spark._wrapped)
        return df

    def run(self):
        result_path = self.spy_api.run(self.output, self.db_name, self.driver_name, self.props)
        return SchemaSpyResult(result_path)

class ScavengerConstraints(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, db_name):
        super().__init__()
        self.output = output
        self.db_name = db_name
        self.table_name = ""
        self.__svg_api = sc._jvm.ScavengerApi

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def infer(self):
        result_path = self.__svg_api.inferConstraints(self.output, self.db_name, self.table_name)
        return SchemaSpyResult(result_path)

class ScavengerRepairMisc(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, db_name, table_name, row_id):
        super().__init__()
        self.db_name = db_name
        self.table_name = table_name
        self.row_id = row_id
        self.target_attr_list = ""
        self.null_ratio = 0.01

        # JVM interfaces for Scavenger APIs
        self.__svg_api = sc._jvm.ScavengerMiscApi

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setTargetAttrList(self, target_attr_list):
        self.target_attr_list = target_attr_list
        return self

    def setNullRatio(self, null_ratio):
        self.null_ratio = null_ratio
        return self

    def injectNull(self):
        if self.table_name is None:
            raise ValueError("`setTableName` should be called before injecting NULL")

        jdf = self.__svg_api.injectNullAt(self.db_name, self.table_name, self.target_attr_list, self.null_ratio)
        df = DataFrame(jdf, self.spark._wrapped)
        return df

    def flatten(self):
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before flattening")

        ret_as_json = json.loads(self.__svg_api.flattenTable(self.db_name, self.table_name, self.row_id))
        return self.spark.table(ret_as_json["flatten"])

class ScavengerRepairModel(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, db_name):
        super().__init__()

        self.constraint_input_path = None
        self.db_name = db_name
        self.table_name = None
        self.row_id = None
        self.black_attr_list = ""
        self.discrete_thres = 80
        self.cnt_value_discretization = False
        self.min_corr_thres = 0.70
        self.domain_threshold_alpha = 0.0
        self.domain_threshold_beta = 0.70
        self.max_attrs_to_compute_domains = 4
        self.attr_stats_sample_ratio = 1.0
        self.training_data_sample_ratio = 1.0
        self.stat_thres_ratio = 0.0
        self.min_features_num = 1
        self.maximal_likelihood_repair_enabled = True
        self.repair_delta = None

        # JVM interfaces for Scavenger APIs
        self.__svg_api = sc._jvm.ScavengerRepairApi

        # Internally used to check elapsed time
        self.__timer_base = None

        # Defines detectors to discover error cells
        self.__detectors = [
            NullErrorDetector(),
            ConstraintErrorDetector()
        ]

        # Temporary views used in repairing processes
        self.__meta_view_names = [
            "discrete_features",
            "gray_cells",
            "repair_base",
            "cell_domain",
            "partial_repaired",
            "repaired",
            "dirty",
            "partial_dirty",
            "weak",
            "train",
            "flatten",
            "dist",
            "score"
        ]

    def misc(self):
        return ScavengerRepairMisc(self.db_name, self.table_name, self.row_id)

    def setConstraints(self, constraint_input_path):
        self.constraint_input_path = constraint_input_path
        return self

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setBlackAttrList(self, black_attr_list):
        self.black_attr_list = black_attr_list
        return self

    def setDomainThresholds(self, alpha, beta):
        self.domain_threshold_alpha = alpha
        self.domain_threshold_beta = beta
        return self

    def setAttrMaxNumToComputeDomains(self, max):
        self.max_attrs_to_compute_domains = max
        return self

    def setMaximalLikelihoodRepairEnabled(self, enabled):
        self.maximal_likelihood_repair_enabled = enabled
        return self

    def setRepairDelta(self, delta):
        self.repair_delta = delta
        return self

    def __temp_name(self, prefix="temp"):
        return "%s_%s" % (prefix, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    def __flatten(self, df):
        temp_view = self.__temp_name()
        df.createOrReplaceTempView(temp_view)
        ret_as_json = json.loads(sc._jvm.ScavengerMiscApi.flattenTable("", temp_view, self.row_id))
        return self.spark.table(ret_as_json["flatten"])

    def __start_spark_jobs(self, name, desc):
        self.spark.sparkContext.setJobGroup(name, name)
        self.__timer_base = time.time()
        self.outputToConsole(desc)

    def __clear_job_group(self):
        # TODO: Uses `SparkContext.clearJobGroup()` instead
        self.spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        self.spark.sparkContext.setLocalProperty("spark.job.description", None)
        self.spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)

    def __end_spark_jobs(self, msg=None):
        id = msg if msg is not None else self.spark.sparkContext.getLocalProperty("spark.jobGroup.id")
        self.outputToConsole("Time to %s is %s(s)" % (id, time.time() - self.__timer_base))
        self.__clear_job_group()

    def __release_resources(self, env):
        for t in list(filter(lambda x: x in self.__meta_view_names, env.values())):
            self.spark.sql("DROP VIEW IF EXISTS %s" % env[t])

    def __check_input(self, env):
        env.update(json.loads(self.__svg_api.checkInputTable(self.db_name, self.table_name, self.row_id)))
        return self.spark.table(env["input_table"]), env["continous_attrs"].split(",")

    def __detect_error_cells(self, env):
        # Initializes defined error detectors with the given env
        for d in self.__detectors:
            d.setup(env)

        error_cells_dfs = [ d.detect() for d in self.__detectors ]

        err_cells = self.__temp_name("gray_cells")
        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        err_cells_df.cache().createOrReplaceTempView(err_cells)
        return err_cells

    def __detect_errors(self, env, error_cells):
        # If `error_cells` provided, just uses it
        if error_cells is not None:
            df = self.spark.table(str(error_cells))
            if not all(c in df.columns for c in (self.row_id, "attribute")):
                raise ValueError("`%s` must have `%s` and `attribute` in columns" % \
                    (str(error_cells), self.row_id))

            self.outputToConsole("Error cells provided by `%s`" % str(error_cells))
            env["gray_cells"] = str(error_cells)
            # We assume that the given error cells are true, so we skip computing error domains
            # with probability because the computational cost is much high.
            self.domain_threshold_beta = 1.0
        else:
            # Applys error detectors to get gray cells
            self.__start_spark_jobs("detect errors",
                "Detecting errors in a table `%s` (%s rows x %s cols)..." % (env["input_table"], env["num_input_rows"], env["num_attrs"]))
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
        self.__start_spark_jobs("preprocess rows",
            "Preprocessing an input table `%s` with rowId=`%s`..." % (self.table_name, self.row_id))
        env.update(json.loads(self.__svg_api.convertToDiscreteFeatures(
            self.db_name, self.table_name, self.row_id,
            self.discrete_thres)))
        self.__end_spark_jobs()

        discrete_ft_df = self.spark.table(env["discrete_features"])
        self.outputToConsole("Valid %s attributes (%s) found and %s continous attributes (%s) included in them" % \
            (len(discrete_ft_df.columns), ",".join(discrete_ft_df.columns), len(continous_attrs), env["continous_attrs"]))

        return discrete_ft_df

    def __analyze_cell_domain(self, env, gray_cells_df, continous_attrs):
        # Checks if attributes are discrete or not, and discretizes continous ones
        discrete_ft_df = self.__preprocess(env, continous_attrs)

        # Computes attribute statistics to calculate domains with posteriori probability
        # based on naïve independence assumptions.
        self.__start_spark_jobs("collect stat rows",
            "Collecting and sampling attribute stats (ratio=%s threshold=%s) before computing error domains..." % \
                (self.attr_stats_sample_ratio, self.stat_thres_ratio))
        env.update(json.loads(self.__svg_api.computeAttrStats(
            env["discrete_features"], env["gray_cells"], self.row_id,
            self.attr_stats_sample_ratio,
            self.stat_thres_ratio)))
        self.__end_spark_jobs("collect %s stat rows" % \
            self.spark.table(env["attr_stats"]).count())

        self.__start_spark_jobs("compute domains",
            "Computing error domains with posteriori probability...")
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
        fix_cells_expr = "if(initValue = domain[0].n, initValue, NULL) val"
        weak_df = cell_domain_df.selectExpr(self.row_id, "attribute", "initValue", fix_cells_expr).cache()
        error_cells_df = weak_df.where("val IS NULL").drop("val").cache()
        weak_df.where("val IS NOT NULL").drop("initValue").cache().createOrReplaceTempView(env["weak"])
        ret_as_json = self.__svg_api.repairAttrsFrom(env["weak"], "", env["repair_base"], self.row_id)
        env["partial_repaired"] = json.loads(ret_as_json)["repaired"]

        logging.info("%d suspicious cells fixed by the computed cell domain `%s` and %d error cells remaining" %
            (self.spark.table(env["weak"]).count(), env["cell_domain"], error_cells_df.count()))

        return error_cells_df

    def __split_clean_and_dirty_rows(self, env, error_cells_df):
        error_rows_df = error_cells_df.selectExpr(self.row_id).distinct().cache()
        fixed_df = self.spark.table(env["partial_repaired"]).join(error_rows_df, self.row_id, "left_anti").cache()
        dirty_df = self.spark.table(env["partial_repaired"]).join(error_rows_df, self.row_id, "left_semi").cache()
        error_attrs = error_cells_df.groupBy("attribute").agg(functions.count("attribute").alias("cnt")).collect()
        assert len(error_attrs) > 0
        return fixed_df, dirty_df, error_attrs

    def __select_training_rows(self, fixed_df):
        # Prepares training data to repair the remaining error cells
        # TODO: Needs more smart sampling, e.g., down-sampling
        train_df = fixed_df.sample(self.training_data_sample_ratio).drop(self.row_id).cache()
        self.outputToConsole("Sampling %s training data (ratio=%s) from %s fixed rows..." % \
            (train_df.count(), self.training_data_sample_ratio, fixed_df.count()))
        return train_df

    def __compute_inference_order(self, env, train_df, error_attrs):
        # Computes domain sizes for training data
        self.__start_spark_jobs("collect training data stats",
            "Collecting training data stats before building ML models...")
        train_df.createOrReplaceTempView(env["train"])
        env.update(json.loads(self.__svg_api.computeDomainSizes(env["train"])))
        self.__end_spark_jobs()

        error_num_map = {}
        for row in error_attrs:
            error_num_map[row.attribute] = row.cnt

        # TODO: Needs to analyze dependencies (e.g., based on graph algorithms) between
        # a target column and the other ones for decideing a inference order.
        #
        # for row in error_attrs:
        #     logging.warning("%s: %s" % (row.attribute, ", ".join(list(map(lambda x: str(x), env["pairwise_attr_stats"][row.attribute])))))
        #     for f, corr in list(map(lambda x: tuple(x), env["pairwise_attr_stats"][row.attribute])):
        #         // Process `f` and `float(corr)` here

        # Sorts target columns by the domain sizes of training data
        # TODO: Needs to think more about feature selection, the order in which models
        # are applied, and corelation of features.
        target_columns = list(map(lambda row: row.attribute, \
            sorted(error_attrs, key=lambda row: int(env["distinct_stats"][row.attribute]), reverse=False)))
        for y in target_columns:
            logging.info("%s: |domain|=%s #errors=%s" % (y, env["distinct_stats"][y], error_num_map[y]))

        return target_columns

    def __build_models(self, env, train_df, error_attrs, continous_attrs):
        # Computes a inference order based on dependencies between `error_attrs` and the others
        target_columns = self.__compute_inference_order(env, train_df, error_attrs)

        # Builds classifiers for the target columns
        self.__start_spark_jobs("build ML models",
            "Building %s ML models to repair the error cells..." % len(target_columns))
        import lightgbm as lgb
        models = {}
        train_pdf = train_df.toPandas()
        # TODO: Needs to think more about this category encoding
        X = train_pdf.applymap(hash)
        excluded_columns = copy.deepcopy(target_columns)
        for index, y in enumerate(target_columns):
            features = [c for c in X.columns if c not in excluded_columns]
            excluded_columns.remove(y)
            if y in continous_attrs:
                reg = lgb.LGBMRegressor()
                reg.fit(X[features], train_pdf[y])
                logging.info("LGBMRegressor[%d]: y(%s)<=X(%s)" % (index, y, ",".join(features)))
                models[y] = (reg, features)
            else:
                clf = lgb.LGBMClassifier()
                clf.fit(X[features], train_pdf[y])
                logging.info("LGBMClassifier[%d]: y(%s)<=X(%s)" % (index, y, ",".join(features)))
                models[y] = (clf, features)

        self.__end_spark_jobs()

        return models, target_columns

    def __repair(self, env, models, target_columns, continous_attrs, dirty_df):
        # Shares all the variables for the learnt models in a Spark cluster
        broadcasted_target_columns = self.spark.sparkContext.broadcast(target_columns)
        broadcasted_continous_attrs = self.spark.sparkContext.broadcast(continous_attrs)
        broadcasted_models = self.spark.sparkContext.broadcast(models)
        broadcasted_maximal_likelihood_repair_enabled = \
            self.spark.sparkContext.broadcast(self.maximal_likelihood_repair_enabled)

        # Sets a grouping key for inference
        num_parallelism = spark.sparkContext.defaultParallelism
        grouping_key = self.__temp_name("__grouping_key")
        dirty_df.createOrReplaceTempView(env["dirty"])
        dirty_df = self.spark.table(env["dirty"]).withColumn(grouping_key, (functions.rand() * functions.lit(num_parallelism)).cast("int"))

        @functions.pandas_udf(dirty_df.schema, functions.PandasUDFType.GROUPED_MAP)
        def repair(pdf):
            target_columns = broadcasted_target_columns.value
            continous_attrs = broadcasted_continous_attrs.value
            models = broadcasted_models.value
            maximal_likelihood_repair_enabled = \
                broadcasted_maximal_likelihood_repair_enabled.value
            rows = []
            for index, row in pdf.iterrows():
                for y in target_columns:
                    (model, features) = models[y]
                    vec = row[features].map(hash)

                    if y in continous_attrs:
                        if np.isnan(row[y]):
                            predicted = model.predict(vec.to_numpy().reshape(1, -1))
                            row[y] = float(predicted[0])
                    else:
                        if row[y] is None:
                            if maximal_likelihood_repair_enabled:
                                predicted = model.predict_proba(vec.to_numpy().reshape(1, -1))
                                dist = { "classes" : model.classes_.tolist(), "probs" : predicted[0].tolist() }
                                row[y] = json.dumps(dist)
                            else:
                                predicted = model.predict(vec.to_numpy().reshape(1, -1))
                                row[y] = predicted[0]

                rows.append(row)

            return pd.DataFrame(rows)

        # Predicts the remaining error cells based on the trained models.
        # TODO: Might need to compare repair costs (cost of an update, c) to
        # the likelihood benefits of the updates (likelihood benefit of an update, l).
        self.__start_spark_jobs("apply the ML models", "Repairing error cells...")
        repaired_df = dirty_df.groupBy(grouping_key).apply(repair).drop(grouping_key).cache()
        repaired_df.write.format("noop").mode("overwrite").save()
        self.__end_spark_jobs()
        return repaired_df

    def __maximal_likelihood_repair(self, env, repaired_df, error_cells_df, continous_attrs):
        # Format a table with probability distribution
        logging.info("Constructing a table (`%s`) for probability distribution..." % env["dist"])
        parse_dist_json_expr = "from_json(val, 'classes array<string>, probs array<double>') dist"
        is_discrete_predicate = "attribute not in (%s)" % ",".join(map(lambda c: "'%s'" % c, continous_attrs))
        to_dist_expr = "arrays_zip(dist.classes, dist.probs) dist"
        to_current_expr = "named_struct('val', initValue, 'prob', " \
            "coalesce(dist.probs[array_position(dist.classes, initValue) - 1], 0.0)) current"
        dist_df = self.__flatten(repaired_df) \
            .join(error_cells_df, [self.row_id, "attribute"], "inner") \
            .where(is_discrete_predicate) \
            .selectExpr(self.row_id, "attribute", "initValue", parse_dist_json_expr) \
            .selectExpr(self.row_id, "attribute", to_current_expr, to_dist_expr)
        dist_df.createOrReplaceTempView(env["dist"])

        # Selects maximal likelihood candidates as correct repairs
        @functions.pandas_udf("double", functions.PandasUDFType.SCALAR)
        def distance(xs, ys):
            import Levenshtein
            dists = [float(Levenshtein.distance(x, y)) for x, y in zip(xs, ys)]
            return pd.Series(dists)

        sorted_dist_expr = "array_sort(dist, (left, right) -> if(left.`1` < right.`1`, 1, -1)) dist"
        maximal_likelihood_repair_expr = "named_struct('val', dist[0].`0`, 'prob', dist[0].`1`) repaired"
        score_expr = "ln(repaired.prob / IF(current.prob > 0.0, current.prob, 1e-6)) * (1.0 / (1.0 + distance)) score"
        score_df = dist_df \
            .selectExpr(self.row_id, "attribute", "current", sorted_dist_expr) \
            .selectExpr(self.row_id, "attribute", "current", maximal_likelihood_repair_expr) \
            .withColumn("distance", distance(col("current.val"), col("repaired.val"))) \
            .selectExpr(self.row_id, "attribute", "repaired.val val", score_expr)

        if self.repair_delta is not None:
            row = score_df.selectExpr("percentile(score, %s) thres" % (float(self.repair_delta) / num_error_cells)).collect()[0]
            score_df = score_df.where("score < %s" % row.thres)
            logging.info("Bounded # of repairs from %s to %s" % (num_error_cells, score_df.count()))

        # Finally, replaces error cells with ones in `score_df`
        repaired_df.createOrReplaceTempView(env["partial_dirty"])
        score_df.createOrReplaceTempView(env["score"])
        env.update(json.loads(self.__svg_api.repairAttrsFrom(env["score"], "", env["partial_dirty"], self.row_id)))
        return self.spark.table(env["repaired"])

    def __run(self, error_cells, detect_errors_only, return_repair_candidates):
        # Env used to repair the given table
        env = {}
        env["row_id"] = self.row_id
        env["constraint_input_path"] = self.constraint_input_path
        env["weak"] = self.__temp_name("weak")
        env["dirty"] = self.__temp_name("dirty")
        env["partial_dirty"] = self.__temp_name("partial_dirty")
        env["train"] = self.__temp_name("train")
        env["dist"] = self.__temp_name("dist")
        env["score"] = self.__temp_name("score")

        # Checks # of input rows and attributes
        input_df, continous_attrs = self.__check_input(env)

        # If no error found, it just returns the given table
        gray_cells_df = self.__detect_errors(env, error_cells)
        if gray_cells_df.count() == 0:
            self.outputToConsole("Any error cells not found, so returns the input as clean cells")
            self.__release_resources(env)
            return input_df

        # Sets NULL to suspicious cells
        repair_base_df = self.__prepare_repair_base(env, gray_cells_df)

        # Selects error cells based on the result of domain analysis
        cell_domain_df = self.__analyze_cell_domain(env, gray_cells_df, continous_attrs)

        # If no error cell found, ready to return a clean table
        error_cells_df = self.__extract_error_cells(env, cell_domain_df, repair_base_df)
        if error_cells_df.count() == 0:
            self.__release_resources(env)
            return input_df

        # If `detect_errors_only` is True, returns found error cells
        if detect_errors_only:
            self.__release_resources(env)
            return error_cells_df

        # Selects rows for training, build models, and repair cells
        fixed_df, dirty_df, error_attrs = self.__split_clean_and_dirty_rows(env, error_cells_df)
        train_df = self.__select_training_rows(fixed_df)

        # Checks if we have the enough number of features for inference
        # TODO: In case of `num_features == 0`, we might be able to select the most accurate and
        # predictable column as a staring feature.
        num_features = len(train_df.columns) - len(error_attrs)
        if num_features < self.min_features_num:
            self.outputToConsole("At least %s features needed to repair error cells, but %s features found" %
                (self.min_features_num, num_features))
            return input_df

        models, target_columns = self.__build_models(env, train_df, error_attrs, continous_attrs)
        repaired_df = self.__repair(env, models, target_columns, continous_attrs, dirty_df)

        # If any discrete target columns and its probability distribution given, computes scores
        # to decide which cells should be repaired to follow the “Maximal Likelihood Repair” problem.
        if self.maximal_likelihood_repair_enabled and len(target_columns) > len(continous_attrs):
            repaired_df = self.__maximal_likelihood_repair(env, repaired_df, error_cells_df, continous_attrs)

        clean_df = fixed_df.union(repaired_df)
        assert clean_df.count() == input_df.count()
        return clean_df

    def run(self, error_cells=None, detect_errors_only=False, return_repair_candidates=False):
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before repairing")

        __start = time.time()
        df = self.__run(error_cells, detect_errors_only, return_repair_candidates)
        self.outputToConsole("!!!Total processing time is %s(s)!!!" % (time.time() - __start))
        return df

class Scavenger(SchemaSpyBase):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(Scavenger, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self):
        super().__init__()
        self.inferType = "default"

    @staticmethod
    def getOrCreate():
        return Scavenger()

    def constraints(self):
        return ScavengerConstraints(self.output, self.db_name)

    def repair(self):
        return ScavengerRepairModel(self.output, self.db_name)

    def setInferType(self, inferType):
        self.inferType = inferType
        return self

    def infer(self):
        result_path = sc._jvm.ScavengerApi.infer(self.output, self.db_name, self.inferType)
        return SchemaSpyResult(result_path)

# Defines singleton variables for SchemaSpy
schemaspy = SchemaSpy.getOrCreate()

# Defines singleton variables for Scavenger
scavenger = Scavenger.getOrCreate()

# This is a method to use SchemaSpy functionality directly
def spySchema(args=""):
    sc._jvm.SchemaSpyApi.run(args)

# Initializes a Spark session
if not sc._jvm.SparkSession.getActiveSession().isDefined():
    spark.sql("SELECT 1")

# Suppress warinig messages in PySpark
warnings.simplefilter('ignore')

# Sets `INFO` to the logging level for debugging
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARN)

# Supresses `WARN` messages in JVM
spark.sparkContext.setLogLevel("ERROR")

# Since 3.0, `spark.sql.crossJoin.enabled` is set to true by default
spark.sql("SET spark.sql.crossJoin.enabled=true")
spark.sql("SET spark.sql.cbo.enabled=true")

# Tunes # shuffle partitions
num_tasks_per_core = 1
num_parallelism = spark.sparkContext.defaultParallelism
spark.sql("SET spark.sql.shuffle.partitions=%s" % (num_parallelism * num_tasks_per_core))

print("Scavenger APIs (version %s) available as 'scavenger'." % ("0.1.0-spark3.0-EXPERIMENTAL"))

