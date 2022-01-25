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
import re
import tempfile
import unittest
import pandas as pd  # type: ignore[import]

from pyspark import SparkConf
from pyspark.sql import Row, functions as func
from pyspark.sql.utils import AnalysisException

from repair.costs import Levenshtein
from repair.errors import ConstraintErrorDetector, DomainValues, NullErrorDetector, RegExErrorDetector
from repair.misc import RepairMisc
from repair.model import FunctionalDepModel, RepairModel, PoorModel
from repair.tests.requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message
from repair.tests.testutils import Eventually, ReusedSQLTestCase, load_testdata


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)  # type: ignore
class RepairModelTests(ReusedSQLTestCase):

    @classmethod
    def conf(cls):
        return SparkConf() \
            .set("spark.master", "local[*]") \
            .set("spark.driver.memory", "4g") \
            .set("spark.jars", os.getenv("REPAIR_API_LIB")) \
            .set("spark.sql.cbo.enabled", "true") \
            .set("spark.sql.statistics.histogram.enabled", "true") \
            .set("spark.sql.statistics.histogram.numBins", "254")

    @classmethod
    def setUpClass(cls):
        super(RepairModelTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads/Defines some test data
        load_testdata(cls.spark, "adult.csv").createOrReplaceTempView("adult")
        load_testdata(cls.spark, "adult_dirty.csv").createOrReplaceTempView("adult_dirty")
        load_testdata(cls.spark, "adult_repair.csv").createOrReplaceTempView("adult_repair")
        load_testdata(cls.spark, "adult_clean.csv").createOrReplaceTempView("adult_clean")

        rows = [
            (1, 0, 1.0, 1.0, 'a'),
            (2, 1, 1.5, 1.5, 'b'),
            (3, 0, 1.4, None, 'b'),
            (4, 1, 1.3, 1.3, 'b'),
            (5, 1, 1.2, 1.1, 'b'),
            (6, 1, 1.1, 1.2, 'b'),
            (7, 0, None, 1.4, 'b'),
            (8, 1, 1.4, 1.0, 'b'),
            (9, 0, 1.2, 1.1, 'b'),
            (10, None, 1.3, 1.2, 'b'),
            (11, 0, 1.0, 1.9, 'b'),
            (12, 0, 1.9, 1.2, 'b'),
            (13, 0, 1.2, 1.3, 'b'),
            (14, 0, 1.8, 1.2, None),
            (15, 0, 1.3, 1.1, 'b'),
            (16, 1, 1.3, 1.0, 'b'),
            (17, 0, 1.3, 1.0, 'b')
        ]
        cls.spark.createDataFrame(rows, ["tid", "v1", "v2", "v3", "v4"]) \
            .createOrReplaceTempView("mixed_input")

        # Define some expected results
        cls.expected_adult_result = cls.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()
        cls.expected_adult_result_without_repaired = cls.spark.table("adult_repair") \
            .selectExpr("tid", "attribute", "current_value") \
            .orderBy("tid", "attribute").collect()

    @classmethod
    def tearDownClass(cls):
        super(ReusedSQLTestCase, cls).tearDownClass()

    def test_invalid_params(self):
        self.assertRaisesRegexp(
            ValueError,
            "`setInput` and `setRowId` should be called before repairing",
            lambda: RepairModel().run())
        self.assertRaisesRegexp(
            ValueError,
            "`setInput` and `setRowId` should be called before repairing",
            lambda: RepairModel().setTableName("dummyTab").run())
        self.assertRaisesRegexp(
            ValueError,
            "`setInput` and `setRowId` should be called before repairing",
            lambda: RepairModel().setInput("dummyTab").run())
        self.assertRaisesRegexp(
            ValueError,
            "Can not specify a database name when input is `DataFrame`",
            lambda: RepairModel().setInput(self.spark.table("adult"))
            .setDbName("default").run())
        self.assertRaisesRegexp(
            ValueError,
            "`setRepairDelta` should be called when enabling maximal likelihood repairing",
            lambda: RepairModel().setTableName("dummyTab").setRowId("dummyId").run(maximal_likelihood_repair=True))
        self.assertRaisesRegexp(
            ValueError,
            "`setRepairDelta` should be called when enabling maximal likelihood repairing",
            lambda: RepairModel().setInput("dummyTab").setRowId("dummyId").run(maximal_likelihood_repair=True))
        self.assertRaisesRegexp(
            ValueError,
            "`setUpdateCostFunction` should be called when enabling maximal likelihood repairing",
            lambda: RepairModel().setInput("dummyTab").setRowId("dummyId")
            .setRepairDelta(3).run(maximal_likelihood_repair=True))
        self.assertRaisesRegexp(
            ValueError,
            "`UpdateCostFunction.targets` cannot be used when enabling maximal likelihood repairing",
            lambda: RepairModel().setInput("dummyTab").setRowId("dummyId")
            .setRepairDelta(3)
            .setUpdateCostFunction(Levenshtein(targets=['non-existent']))
            .run(maximal_likelihood_repair=True))
        self.assertRaisesRegexp(
            ValueError,
            "`attrs` should have at least one attribute",
            lambda: RepairModel().setTargets([]))
        self.assertRaisesRegexp(
            ValueError,
            "`thres` should be bigger than 1, got 0",
            lambda: RepairModel().setDiscreteThreshold(0))
        self.assertRaisesRegexp(
            ValueError,
            "Can not specify a database name when input is `DataFrame`",
            lambda: RepairModel().setInput(self.spark.range(0)).setDbName('db'))
        self.assertRaisesRegexp(
            ValueError,
            "`table_name` should have at least character",
            lambda: RepairModel().setTableName(''))
        self.assertRaisesRegexp(
            ValueError,
            "`table_name` should have at least character",
            lambda: RepairModel().setInput(''))
        self.assertRaisesRegexp(
            ValueError,
            "`row_id` should have at least character",
            lambda: RepairModel().setRowId(''))
        self.assertRaisesRegexp(
            ValueError,
            "`thres` should be bigger than 1, got 0",
            lambda: RepairModel().setDiscreteThreshold(0))
        self.assertRaisesRegexp(
            ValueError,
            "Repair delta should be positive, got -1",
            lambda: RepairModel().setRepairDelta(-1))
        self.assertRaisesRegexp(
            ValueError,
            "`error_cells` should have at least character",
            lambda: RepairModel().setErrorCells(''))

    def test_exclusive_params(self):
        def _assert_exclusive_params(func):
            self.assertRaisesRegexp(ValueError, "cannot be set to true simultaneously", func)
        test_model = RepairModel()
        api = test_model.setTableName("dummyTab").setRowId("dummyId")
        _assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, compute_repair_candidate_prob=True))
        _assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, repair_data=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, repair_data=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_repair_prob=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_repair_score=True))

    def test_argtype_check(self):
        self.assertRaisesRegexp(
            TypeError,
            "`db_name` should be provided as str, got int",
            lambda: RepairModel().setDbName(1))
        self.assertRaisesRegexp(
            TypeError,
            "`table_name` should be provided as str, got int",
            lambda: RepairModel().setTableName(1))
        self.assertRaisesRegexp(
            TypeError,
            "`thres` should be provided as int, got str",
            lambda: RepairModel().setDiscreteThreshold("a"))
        self.assertRaisesRegexp(
            TypeError,
            "`input` should be provided as str/DataFrame, got int",
            lambda: RepairModel().setInput(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`attrs` should be provided as list[str], got int"),
            lambda: RepairModel().setTargets(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`attrs` should be provided as list[str], got int in elements"),
            lambda: RepairModel().setTargets(["a", 1]))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`detectors` should be provided as list[ErrorDetector], got int"),
            lambda: RepairModel().setErrorDetectors(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`detectors` should be provided as list[ErrorDetector], got int in elements"),
            lambda: RepairModel().setErrorDetectors([1]))
        self.assertRaisesRegexp(
            TypeError,
            "`cf` should be provided as UpdateCostFunction, got int",
            lambda: RepairModel().setUpdateCostFunction(1))
        self.assertRaisesRegexp(
            TypeError,
            "`cf` should be provided as UpdateCostFunction, got list",
            lambda: RepairModel().setUpdateCostFunction([1]))

    def test_invalid_running_modes(self):
        test_model = RepairModel() \
            .setTableName("mixed_input") \
            .setRowId("tid") \
            .setRepairDelta(1) \
            .setUpdateCostFunction(Levenshtein())
        self.assertRaisesRegexp(
            ValueError,
            "Cannot enable the maximal likelihood repair mode when continous attributes found",
            lambda: test_model.run(maximal_likelihood_repair=True))

        test_model = RepairModel() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setRepairByRules(True) \
            .setUpdateCostFunction(Levenshtein()) \
            .setRepairDelta(3) \
            .option("model.rule.repair_by_nearest_values.disabled", "")
        expected_error_msg = "Cannot repair data by nearest values when enabling `maximal_likelihood_repair`, " \
            "`compute_repair_candidate_prob`, `compute_repair_prob`, or `compute_repair_score`"
        self.assertRaisesRegexp(
            ValueError,
            expected_error_msg,
            lambda: test_model.run(maximal_likelihood_repair=True))
        self.assertRaisesRegexp(
            ValueError,
            expected_error_msg,
            lambda: test_model.run(compute_repair_candidate_prob=True))
        self.assertRaisesRegexp(
            ValueError,
            expected_error_msg,
            lambda: test_model.run(compute_repair_prob=True))
        self.assertRaisesRegexp(
            ValueError,
            expected_error_msg,
            lambda: test_model.run(compute_repair_score=True))

    # TODO: We fix a seed for building a repair model, but inferred values fluctuate run-by-run.
    # So, to avoid it, we set 1 to `hp.max_evals` for now.
    def _build_model(self):
        return RepairModel() \
            .setErrorDetectors([NullErrorDetector()]) \
            .option("model.hp.max_evals", "1")

    def test_options(self):
        self.assertRaisesRegexp(
            ValueError,
            "Non-existent key specified: key=non-existent",
            lambda: RepairModel().option('non-existent', '1'))

        test_option_keys = [
            ('error.pairwise_attr_corr_threshold', '1.0'),
            ('error.domain_threshold_alpha', '0.0'),
            ('error.domain_threshold_beta', '0.7'),
            ('error.max_attrs_to_compute_domains', '4'),
            ('error.attr_stat_sample_ratio', '1.0'),
            ('error.freq_attr_stat_threshold', '0.0'),
            ('model.max_training_row_num', '100000'),
            ('model.max_training_column_num', '65536'),
            ('model.small_domain_threshold', '12'),
            ('model.rule.repair_by_nearest_values.disabled', '1'),
            ('model.rule.merge_threshold', '2.0'),
            ('model.rule.repair_by_regex.disabled', ''),
            ('model.rule.repair_by_functional_deps.disabled', ''),
            ('model.rule.max_domain_size', '1000'),
            ('repair.pmf.cost_weight', '0.1'),
            ('repair.pmf.prob_threshold', '0.0'),
            ('repair.pmf.prob_top_k', '80'),
            ('model.lgb.boosting_type', 'gbdt'),
            ('model.lgb.class_weight', 'balanced'),
            ('model.lgb.learning_rate', '0.01'),
            ('model.lgb.max_depth', '7'),
            ('model.lgb.max_bin', '255'),
            ('model.lgb.reg_alpha', '0.0'),
            ('model.lgb.min_split_gain', '0.0'),
            ('model.lgb.n_estimators', '300'),
            ('model.lgb.importance_type', 'gain'),
            ('model.cv.n_splits', '3'),
            ('model.hp.timeout', '0'),
            ('model.hp.max_evals', '10000000'),
            ('model.hp.no_progress_loss', '50')
        ]
        for key, value in test_option_keys:
            try:
                RepairModel().option(key, value)
            except Exception as e:
                self.assertTrue(False, msg=str(e))

    def test_invalid_internal_options(self):
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid")
        self.assertRaisesRegexp(
            ValueError,
            'Failed to cast "invalid" into float data: key=error.pairwise_attr_corr_threshold',
            lambda: test_model.option('error.pairwise_attr_corr_threshold', 'invalid').run())

    def test_multiple_run(self):
        # Checks if auto-generated views are dropped finally
        current_view_nums = self.spark.sql("SHOW VIEWS").count()

        def _test_basic():
            test_model = self._build_model() \
                .setTableName("adult") \
                .setRowId("tid") \
                .option('error.pairwise_attr_corr_threshold', '1.0') \
                .option('error.domain_threshold_alpha', '0.0') \
                .option('error.domain_threshold_beta', '0.70') \
                .option('error.max_attrs_to_compute_domains', '4') \
                .option('error.attr_stat_sample_ratio', '1.0') \
                .option('error.freq_attr_stat_threshold', '0.0') \
                .option('model.max_training_row_num', '10000') \
                .option('model.max_training_column_num', '65536') \
                .option('model.small_domain_threshold', '12') \
                .option('model.lgb.boosting_type', 'gbdt') \
                .option('model.lgb.class_weight', 'balanced') \
                .option('model.lgb.learning_rate', '0.01') \
                .option('model.lgb.max_depth', '7') \
                .option('model.lgb.max_bin', '255') \
                .option('model.lgb.reg_alpha', '0.0') \
                .option('model.lgb.min_split_gain', '0.0') \
                .option('model.lgb.n_estimators', '300') \
                .option('model.lgb.importance_type', 'gain') \
                .option('model.cv.n_splits', '3') \
                .option('model.hp.timeout', '0') \
                .option('model.hp.max_evals', '1') \
                .option('model.hp.no_progress_loss', '50')
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                self.expected_adult_result)

        _test_basic()  # first run
        _test_basic()  # second run

        self.assertEqual(
            self.spark.sql("SHOW VIEWS").count(),
            current_view_nums)

    def test_parallel_stat_training(self):
        df = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setParallelStatTrainingEnabled(True) \
            .run()
        self.assertEqual(
            df.orderBy("tid", "attribute").collect(),
            self.expected_adult_result)

    def test_table_input(self):
        with self.table("adult_table"):
            # Tests for `setDbName`
            self.spark.table("adult").write.mode("overwrite").saveAsTable("adult_table")
            test_model = self._build_model() \
                .setDbName("default") \
                .setTableName("adult_table") \
                .setRowId("tid")

            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                self.expected_adult_result)

    def test_input_overwrite(self):
        with self.table("adult_table"):
            # Tests for input overwrite case ("default.adult_table" -> "adult")
            self.spark.table("adult").write.mode("overwrite").saveAsTable("adult_table")
            test_model = self._build_model() \
                .setDbName("default") \
                .setTableName("adult_table") \
                .setInput(self.spark.table("adult")) \
                .setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                self.expected_adult_result)

    def test_setInput(self):
        def _test_setInput(input):
            test_model = self._build_model().setInput(input).setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                self.expected_adult_result)

        _test_setInput("adult")
        _test_setInput(self.spark.table("adult"))

    def test_setTargets(self):
        error_cells_df = expected_result = self.spark.table("adult_repair") \
            .selectExpr("tid", "attribute") \
            .orderBy("tid", "attribute")

        def _test_setTargets(targets):
            actual_result = self._build_model() \
                .setInput("adult") \
                .setRowId("tid") \
                .setTargets(targets) \
                .run() \
                .selectExpr("tid", "attribute") \
                .orderBy("tid", "attribute")
            expected_result = error_cells_df \
                .where("attribute IN ({})".format(",".join(map(lambda x: f"'{x}'", targets)))) \
                .orderBy("tid", "attribute") \
                .collect()
            self.assertEqual(
                actual_result.collect(),
                expected_result)

        _test_setTargets(["Sex"])
        _test_setTargets(["Sex", "Income"])
        _test_setTargets(["Age", "Sex"])
        _test_setTargets(["Non-Existent", "Age"])

        self.assertRaisesRegexp(
            ValueError,
            'Target attributes not found in adult: Non-Existent',
            lambda: self._build_model().setInput("adult").setRowId("tid").setTargets(["Non-Existent"]).run())

    def test_setErrorCells(self):
        self.assertRaisesRegexp(
            ValueError,
            '`setRowId` should be called before specifying error cells',
            lambda: self._build_model().setErrorCells('adult_dirty').setInput("adult").setRowId("tid").run())

        self.assertRaisesRegexp(
            ValueError,
            'Error cells should have `tid` and `attribute` in columns',
            lambda: self._build_model().setInput("adult").setRowId("tid").setErrorCells('adult').run())

        def _test_setErrorCells(error_cells):
            test_model = self._build_model() \
                .setTableName("adult") \
                .setRowId("tid") \
                .setErrorCells(error_cells)
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                self.expected_adult_result)

        _test_setErrorCells("adult_dirty")
        _test_setErrorCells(self.spark.table("adult_dirty"))
        _test_setErrorCells(self.spark.table("adult_dirty").withColumn('unrelated', func.expr('1')))

    def test_setErrorCells_and_detect_errors_only(self):
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setErrorCells(self.spark.table("adult_dirty").withColumn('unrelated', func.expr('1')))
        self.assertEqual(
            test_model.run(detect_errors_only=True).orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute='Sex', current_value=None),
                Row(tid=5, attribute='Age', current_value=None),
                Row(tid=5, attribute='Income', current_value=None),
                Row(tid=7, attribute='Sex', current_value=None),
                Row(tid=12, attribute='Age', current_value=None),
                Row(tid=12, attribute='Sex', current_value=None),
                Row(tid=16, attribute='Income', current_value=None)])

    def _assert_adult_without_repaired(self, test_model):
        def _test(df):
            self.assertEqual(
                df.selectExpr("tid", "attribute", "current_value").orderBy("tid", "attribute").collect(),
                self.expected_adult_result_without_repaired)
        _test(test_model.setParallelStatTrainingEnabled(False).run())
        _test(test_model.setParallelStatTrainingEnabled(True).run())

    def test_error_cells_having_no_existent_attribute(self):
        error_cells = [
            Row(tid=1, attribute="NoExistent"),
            Row(tid=5, attribute="Income"),
            Row(tid=16, attribute="Income")
        ]
        error_cells_df = self.spark.createDataFrame(
            error_cells, schema="tid STRING, attribute STRING")
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setErrorCells(error_cells_df)
        self.assertEqual(
            test_model.run().orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income", current_value=None, repaired="MoreThan50K"),
                Row(tid=16, attribute="Income", current_value=None, repaired="MoreThan50K")])

    def test_detect_errors_only(self):
        # Tests for `NullErrorDetector`
        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(),
            self.expected_adult_result_without_repaired)

        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Sex", "Age", "Income"]) \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(),
            self.expected_adult_result_without_repaired)

        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Sex"]) \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex", current_value=None),
                Row(tid=7, attribute="Sex", current_value=None),
                Row(tid=12, attribute="Sex", current_value=None)])

        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Age", "Income"]) \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Age", current_value=None),
                Row(tid=5, attribute="Income", current_value=None),
                Row(tid=12, attribute="Age", current_value=None),
                Row(tid=16, attribute="Income", current_value=None)])

        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Unknown", "Age"]) \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Age", current_value=None),
                Row(tid=12, attribute="Age", current_value=None)])

        # Tests for `DomainValues`
        error_detectors = [
            DomainValues("Country", ["United-States"]),
            DomainValues("Income", ["LessThan50K", "MoreThan50K"])
        ]
        domain_value_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            domain_value_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income", current_value=None),
                Row(tid=7, attribute="Country", current_value="India"),
                Row(tid=16, attribute="Income", current_value=None),
                Row(tid=19, attribute="Country", current_value="Iran")])

        # Tests for `RegExErrorDetector`
        error_detectors = [
            RegExErrorDetector("Country", "United-States"),
            RegExErrorDetector("Relationship", "(Husband|Own-child|Not-in-family)")
        ]
        regex_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Country", "Relationship"]) \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            regex_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country", current_value="India"),
                Row(tid=14, attribute="Relationship", current_value="Unmarried"),
                Row(tid=16, attribute="Relationship", current_value="Unmarried"),
                Row(tid=19, attribute="Country", current_value="Iran")])

        # Tests for `ConstraintErrorDetector`
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        error_detectors = [
            ConstraintErrorDetector(constraint_path)
        ]
        constraint_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Sex", "Relationship"]) \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            constraint_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship", current_value="Husband"),
                Row(tid=4, attribute="Sex", current_value="Female"),
                Row(tid=11, attribute="Relationship", current_value="Husband"),
                Row(tid=11, attribute="Sex", current_value="Female")])
        error_detectors = [
            ConstraintErrorDetector(constraint_path, targets=["Sex"])
        ]
        constraint_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Sex", "Relationship"]) \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            constraint_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Sex", current_value="Female"),
                Row(tid=11, attribute="Sex", current_value="Female")])

        # Model reuse tests for `ConstraintErrorDetector`
        error_detectors = [
            ConstraintErrorDetector(constraint_path, targets=["Sex", "Relationship"])
        ]
        test_model_with_constraints = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setTargets(["Sex"]) \
            .setErrorDetectors(error_detectors)
        constraint_errors = test_model_with_constraints.run(detect_errors_only=True)
        self.assertEqual(
            constraint_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Sex", current_value="Female"),
                Row(tid=11, attribute="Sex", current_value="Female")])
        constraint_errors = test_model_with_constraints.setTargets(["Relationship"]).run(detect_errors_only=True)
        self.assertEqual(
            constraint_errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship", current_value="Husband"),
                Row(tid=11, attribute="Relationship", current_value="Husband")])

    def test_DomainValues_against_continous_values(self):
        with self.tempView("inputView"):
            rows = [
                (1, 1.0, 1.0, 1.0),
                (2, 1.1, 1.1, 1.1),
                (3, 1.0, 1.0, None),
                (4, 1.1, 1.0, 1.0),
                (5, 1.1, 1.1, 1.1),
                (6, 1.0, 1.0, None)
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y", "z"]) \
                .createOrReplaceTempView("inputView")

            error_detectors = [
                DomainValues('x', autofill=True, min_count_thres=2),
                DomainValues('y', autofill=True, min_count_thres=2),
                DomainValues('z', autofill=True, min_count_thres=2),
                NullErrorDetector()
            ]
            domain_value_errors = self._build_model() \
                .setInput("inputView") \
                .setRowId("tid") \
                .setErrorDetectors(error_detectors) \
                .run(detect_errors_only=True)
            self.assertEqual(
                domain_value_errors.orderBy("tid", "attribute").collect(), [
                    Row(tid=3, attribute="z", current_value=None),
                    Row(tid=6, attribute="z", current_value=None)])

    def test_repair_data(self):
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid")
        expected_result = self.spark.table("adult_clean") \
            .orderBy("tid").collect()
        self.assertEqual(
            test_model.run(repair_data=True).orderBy("tid").collect(),
            expected_result)

    def test_escaped_column_names(self):
        with self.tempView("inputView1", "inputView2"):
            rows = [
                (1, "1", None, 1.0),
                (2, None, "test-2", 2.0),
                (3, "1", "test-1", 1.0),
                (4, "2", "test-2", 2.0),
                (5, "2", "test-2", 1.0),
                (6, "1", "test-1", 1.0)
            ]
            self.spark.createDataFrame(rows, ["t i d", "x x", "y y", "z z"]) \
                .createOrReplaceTempView("inputView1")

            test_model = self._build_model() \
                .setTableName("inputView1") \
                .setRowId("t i d") \
                .setDiscreteThreshold(10)
            self.assertEqual(
                test_model.run().orderBy("t i d", "attribute").collect(), [
                    Row(1, "y y", None, 'test-1'),
                    Row(2, "x x", None, '2')])
            df = test_model.run(compute_repair_candidate_prob=True).selectExpr('`t i d`', 'attribute')
            self.assertEqual(
                df.orderBy('t i d', 'attribute').collect(), [
                    Row(1, "y y"),
                    Row(2, "x x")])
            df = test_model.run(compute_repair_prob=True).selectExpr('`t i d`', 'attribute')
            self.assertEqual(
                df.orderBy('t i d', 'attribute').collect(), [
                    Row(1, "y y"),
                    Row(2, "x x")])
            self.assertEqual(
                test_model.run(repair_data=True).where('`t i d` IN (1, 2)').orderBy("t i d").collect(), [
                    Row(1, '1', 'test-1', 1.0),
                    Row(2, '2', 'test-2', 2.0)])

            self.spark.table('inputView1').selectExpr('`t i d`', '`x x`', '`y y`') \
                .createOrReplaceTempView("inputView2")
            test_model = self._build_model() \
                .setTableName("inputView2") \
                .setRowId("t i d") \
                .setDiscreteThreshold(10) \
                .setUpdateCostFunction(Levenshtein()) \
                .setRepairDelta(3)
            df = test_model \
                .run(compute_repair_score=True) \
                .selectExpr('`t i d`', 'attribute')
            self.assertEqual(
                df.orderBy('t i d', 'attribute').collect(), [
                    Row(1, "y y"),
                    Row(2, "x x")])

    def test_unsupported_types(self):
        with self.tempView("inputView"):
            self.spark.range(1).selectExpr("id tid", "1 AS x", "CAST('2021-08-01' AS DATE) y") \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaisesRegexp(
                AnalysisException,
                'Supported types are tinyint,float,smallint,string,double,int,bigint, but unsupported ones found: date',
                lambda: test_model.run())

    def test_max_training_column_num(self):
        df = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setDiscreteThreshold(5) \
            .option("model.max_training_column_num", "2") \
            .run()
        self.assertEqual(
            df.orderBy("tid", "attribute").collect(),
            self.expected_adult_result)

    def test_table_has_no_enough_columns(self):
        with self.tempView("inputView"):
            rows = [
                (1, None),
                (2, "test-1"),
                (3, "test-1")
            ]
            self.spark.createDataFrame(rows, ["tid", "x"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaisesRegexp(
                AnalysisException,
                re.escape("A least three columns (`tid` columns + two more ones) in table 'inputView'"),
                lambda: test_model.run())

    def test_rowid_uniqueness(self):
        with self.tempView("inputView"):
            rows = [
                (1, 1, None),
                (1, 1, "test-1"),
                (1, 2, "test-1")
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaisesRegexp(
                AnalysisException,
                re.escape("Uniqueness does not hold in column 'tid' of table 'inputView' "
                          "(# of distinct 'tid': 1, # of rows: 3)"),
                lambda: test_model.run())

    def test_no_valid_discrete_feature_exists_1(self):
        with self.tempView("inputView"):
            rows = [
                (1, "1", None),
                (2, "1", None),
                (3, "1", "test-1"),
                (4, "1", "test-1"),
                (5, "1", "test-1"),
                (6, "1", None)
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaisesRegexp(
                ValueError,
                "At least one valid discretizable feature is needed to repair error cells",
                lambda: test_model.run())

    def test_no_valid_discrete_feature_exists_2(self):
        with self.tempView("inputView"):
            rows = [
                (1, "1", None),
                (2, "2", "test-2"),
                (3, "3", "test-3"),
                (4, "4", "test-4"),
                (5, "5", "test-5"),
                (6, "6", "test-6")
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid") \
                .setDiscreteThreshold(3)
            self.assertRaisesRegexp(
                ValueError,
                "At least one valid discretizable feature is needed to repair error cells",
                lambda: test_model.run(detect_errors_only=False))
            self.assertEqual(
                test_model.run(detect_errors_only=True).collect(), [
                    Row(tid=1, attribute="y", current_value=None)])

    def test_no_repairable_cell_exists(self):
        with self.tempView("inputView"):
            rows = [
                (1, "1", None),
                (2, "2", None),
                (3, "1", "test-1"),
                (4, "1", "test-1"),
                (5, "1", "test-1"),
                (6, "1", None)
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaisesRegexp(
                ValueError,
                "At least one valid discretizable feature is needed to repair error cells, but no such feature found",
                lambda: test_model.run(detect_errors_only=False))
            self.assertEqual(
                test_model.run(detect_errors_only=True).orderBy("tid", "attribute").collect(), [
                    Row(tid=1, attribute="y", current_value=None),
                    Row(tid=2, attribute="y", current_value=None),
                    Row(tid=6, attribute="y", current_value=None)])

    def test_regressor_model(self):
        with self.tempView("inputView"):
            rows = [
                (1, 1.0, 1.0, 1.0),
                (2, 1.5, 1.5, 1.5),
                (3, 1.4, 1.4, None),
                (4, 1.3, 1.3, 1.3),
                (5, 1.1, 1.1, 1.1),
                (6, 1.2, 1.2, None)
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y", "z"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")

            df = test_model.run().orderBy("tid", "attribute")
            self.assertEqual(
                df.selectExpr("tid", "attribute", "current_value").collect(), [
                    Row(tid=3, attribute="z", current_value=None),
                    Row(tid=6, attribute="z", current_value=None)])

            rows = df.selectExpr("repaired").collect()
            self.assertTrue(rows[0].repaired is not None)
            self.assertTrue(rows[1].repaired is not None)

    def test_repair_by_functional_deps(self):
        with self.tempView("inputView", "errorCells"):
            rows = [
                (1, "1", "test-1"),
                (2, "2", "test-2"),
                (3, "1", None),
                (4, "2", "test-2"),
                (5, "2", None),
                (6, "3", None)
            ]
            self.spark.createDataFrame(rows, ["tid", "x", "y"]) \
                .createOrReplaceTempView("inputView")

            self.spark.createDataFrame([(3, "y"), (5, "y"), (6, "y")], ["tid", "attribute"]) \
                .createOrReplaceTempView("errorCells")

            with tempfile.NamedTemporaryFile("w+t") as f:
                # Creates a file for constraints
                f.write("t1&t2&EQ(t1.x,t2.x)&IQ(t1.y,t2.y)")
                f.flush()

                error_detectors = [
                    NullErrorDetector(),
                    ConstraintErrorDetector(f.name)
                ]
                test_model = self._build_model() \
                    .setTableName("inputView") \
                    .setRowId("tid") \
                    .setErrorCells("errorCells") \
                    .setErrorDetectors(error_detectors) \
                    .setRepairByRules(True) \
                    .option('model.rule.max_domain_size', '1000')
                self.assertEqual(
                    test_model.run().orderBy("tid", "attribute").collect(), [
                        Row(tid=3, attribute="y", current_value=None, repaired="test-1"),
                        Row(tid=5, attribute="y", current_value=None, repaired="test-2"),
                        Row(tid=6, attribute="y", current_value=None, repaired=None)])

    def test_repair_by_nearest_values(self):
        with self.tempView("inputView", "errorCells"):
            rows = [
                (1, "100%", 100, "a", 1.0),
                (3, "32%", 101, "b", 1.1),
                (4, "1xx%", 1, "a", 1.3),
                (5, "100x", 2, "b", 0.6),
                (6, "12x", 300, "a", 0.8)
            ]
            self.spark.createDataFrame(rows, ["tid", "v0", "v1", "v2", "v3"]) \
                .createOrReplaceTempView("inputView")

            error_cells = [(4, "v0"), (5, "v0"), (6, "v0"), (3, "v1"), (5, "v1"), (6, "v1"), (5, "v2")]
            self.spark.createDataFrame(error_cells, ["tid", "attribute"]) \
                .createOrReplaceTempView("errorCells")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid") \
                .setErrorCells("errorCells") \
                .setRepairByRules(True) \
                .setUpdateCostFunction(Levenshtein(targets=["v0", "v1"])) \
                .option("model.rule.repair_by_nearest_values.disabled", "") \
                .option("model.rule.merge_threshold", "2.0")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(), [
                    Row(tid=3, attribute='v1', current_value='101', repaired='100'),
                    Row(tid=4, attribute='v0', current_value='1xx%', repaired='100%'),
                    Row(tid=5, attribute='v0', current_value='100x', repaired='100%'),
                    Row(tid=5, attribute='v1', current_value='2', repaired='1'),
                    Row(tid=5, attribute='v2', current_value='b', repaired='a'),
                    Row(tid=6, attribute='v0', current_value='12x', repaired='32%'),
                    Row(tid=6, attribute='v1', current_value='300', repaired='100')])

            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid") \
                .setTargets(["v0", "v1"]) \
                .setErrorCells("errorCells") \
                .setRepairByRules(True) \
                .setUpdateCostFunction(Levenshtein(targets=["v0", "v1"])) \
                .option("model.rule.repair_by_nearest_values.disabled", "") \
                .option("model.rule.merge_threshold", "2.0")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(), [
                    Row(tid=3, attribute='v1', current_value='101', repaired='100'),
                    Row(tid=4, attribute='v0', current_value='1xx%', repaired='100%'),
                    Row(tid=5, attribute='v0', current_value='100x', repaired='100%'),
                    Row(tid=5, attribute='v1', current_value='2', repaired='1'),
                    Row(tid=6, attribute='v0', current_value='12x', repaired='32%'),
                    Row(tid=6, attribute='v1', current_value='300', repaired='100')])

    def test_repair_updates(self):
        expected_result = self.spark.table("adult_clean") \
            .orderBy("tid").collect()
        repair_updates_df = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .run()

        with self.tempView("repair_updates_view"):
            repair_updates_df.createOrReplaceTempView("repair_updates_view")
            df = RepairMisc() \
                .option("repair_updates", "repair_updates_view") \
                .option("table_name", "adult") \
                .option("row_id", "tid") \
                .repair()
            self.assertEqual(
                df.orderBy("tid").collect(),
                expected_result)

    def _check_adult_repair_prob_and_score(self, df, expected_schema):
        self.assertEqual(
            df.schema.simpleString(),
            expected_schema)
        self.assertEqual(
            df.selectExpr("tid", "attribute", "current_value").orderBy("tid", "attribute").collect(),
            self.expected_adult_result_without_repaired)

    def test_compute_repair_candidate_prob(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .option('repair.pmf.cost_weight', '0.1') \
            .option('repair.pmf.prob_threshold', '0.0') \
            .option('repair.pmf.prob_top_k', '80') \
            .run(compute_repair_candidate_prob=True)

        self._check_adult_repair_prob_and_score(
            repaired_df,
            "struct<tid:int,attribute:string,current_value:string,"
            "pmf:array<struct<class:string,prob:double>>>")

    def test_compute_weighted_probs_for_target_attributes(self):
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        error_detectors = [
            ConstraintErrorDetector(constraint_path)
        ]
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setTargets(["Sex", "Relationship"]) \
            .setErrorDetectors(error_detectors) \
            .option("model.hp.max_evals", "1000") \
            .option("model.hp.no_progress_loss", "150")

        base_rows = test_model \
            .run(compute_repair_candidate_prob=True) \
            .selectExpr('tid', 'attribute', 'pmf[0].class value', 'pmf[0].prob prob') \
            .orderBy("tid", "attribute") \
            .collect()

        weighted_prob_rows = test_model \
            .setUpdateCostFunction(Levenshtein(targets=["Sex"])) \
            .option("repair.pmf.cost_weight", "100000000.0") \
            .run(compute_repair_candidate_prob=True) \
            .selectExpr('tid', 'attribute', 'pmf[0].class value', 'pmf[0].prob prob') \
            .orderBy("tid", "attribute") \
            .collect()

        for r1, r2 in zip(base_rows, weighted_prob_rows):
            self.assertEqual(r1.tid, r1.tid)
            self.assertEqual(r1.attribute, r1.attribute)
            self.assertEqual(r1.value, r1.value)
            if r1.attribute == 'Sex':
                self.assertLess(r1.prob, 0.95)
                self.assertGreater(r2.prob, 0.9999)
            else:  # 'Relationship' case
                self.assertLess(r1.prob, 0.95)
                self.assertLess(r2.prob, 0.95)

    def test_compute_repair_prob(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .run(compute_repair_prob=True)

        self._check_adult_repair_prob_and_score(
            repaired_df,
            "struct<tid:int,attribute:string,current_value:string,"
            "repaired:string,prob:double>")

    def test_compute_repair_score(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setUpdateCostFunction(Levenshtein()) \
            .setRepairDelta(1) \
            .run(compute_repair_score=True)

        self._check_adult_repair_prob_and_score(
            repaired_df,
            "struct<tid:int,attribute:string,current_value:string,"
            "repaired:string,score:double>")

    def test_maximal_likelihood_repair(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setUpdateCostFunction(Levenshtein()) \
            .setRepairDelta(3) \
            .run(maximal_likelihood_repair=True)
        self.assertEqual(
            repaired_df.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex", current_value=None, repaired="Male"),
                Row(tid=7, attribute="Sex", current_value=None, repaired="Male"),
                Row(tid=12, attribute="Sex", current_value=None, repaired="Male")])

    def test_compute_repair_prob_for_continouos_values(self):
        def run_test(f=lambda m: m):
            def _test(df, expected_schema):
                self.assertEqual(
                    df.schema.simpleString(),
                    expected_schema)
                self.assertEqual(
                    df.selectExpr("tid", "attribute", "current_value").orderBy("tid", "attribute").collect(), [
                        Row(tid=3, attribute="v3", current_value=None),
                        Row(tid=7, attribute="v2", current_value=None),
                        Row(tid=10, attribute="v1", current_value=None),
                        Row(tid=14, attribute="v4", current_value=None)])

            test_model = f(self._build_model().setTableName("mixed_input").setRowId("tid"))
            _test(test_model.run(compute_repair_candidate_prob=True),
                  "struct<tid:bigint,attribute:string,current_value:string,"
                  "pmf:array<struct<class:string,prob:double>>>")
            _test(test_model.run(compute_repair_prob=True),
                  "struct<tid:bigint,attribute:string,current_value:string,"
                  "repaired:string,prob:double>")

        run_test()
        run_test(lambda m: m.setUpdateCostFunction(Levenshtein()))

    def test_integer_input(self):
        with self.tempView("int_input"):
            rows = [
                (1, 1, 1, 3, 0),
                (2, 2, None, 2, 1),
                (3, 3, 2, 2, 0),
                (4, 2, 2, 3, 1),
                (5, None, 1, 3, 0),
                (6, 2, 2, 3, 0),
                (7, 3, 1, None, 0),
                (8, 2, 1, 2, 1),
                (9, 1, 1, 2, None)
            ]
            self.spark.createDataFrame(rows, "tid: int, v1: byte, v2: short, v3: int, v4: long") \
                .createOrReplaceTempView("int_input")

            df = test_model = self._build_model() \
                .setTableName("int_input") \
                .setRowId("tid") \
                .run()
            self.assertEqual(
                df.orderBy("tid", "attribute").collect(), [
                    Row(tid=2, attribute="v2", current_value=None, repaired="2"),
                    Row(tid=5, attribute="v1", current_value=None, repaired="2"),
                    Row(tid=7, attribute="v3", current_value=None, repaired="2"),
                    Row(tid=9, attribute="v4", current_value=None, repaired="1")])

    def test_rule_based_model(self):
        model = FunctionalDepModel("x", {1: "test-1", 2: "test-1", 3: "test-2"})
        pdf = pd.DataFrame([[3], [1], [2], [4]], columns=["x"])
        self.assertEqual(model.classes_.tolist(), ["test-1", "test-2"])
        self.assertEqual(model.predict(pdf), ["test-2", "test-1", "test-1", None])
        pmf = model.predict_proba(pdf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf[0].tolist(), [0.0, 1.0])
        self.assertEqual(pmf[1].tolist(), [1.0, 0.0])
        self.assertEqual(pmf[2].tolist(), [1.0, 0.0])
        self.assertIsNone(pmf[3])

    def test_PoorModel(self):
        model = PoorModel(None)
        pdf = pd.DataFrame([[3], [1], [2], [4]], columns=["x"])
        self.assertEqual(model.classes_.tolist(), [None])
        self.assertEqual(model.predict(pdf), [None, None, None, None])
        pmf = model.predict_proba(pdf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf[0].tolist(), [1.0])
        self.assertEqual(pmf[1].tolist(), [1.0])
        self.assertEqual(pmf[2].tolist(), [1.0])
        self.assertEqual(pmf[3].tolist(), [1.0])

        model = PoorModel("test")
        pdf = pd.DataFrame([[3], [1], [2], [4]], columns=["x"])
        self.assertEqual(model.classes_.tolist(), ["test"])
        self.assertEqual(model.predict(pdf), ["test", "test", "test", "test"])
        pmf = model.predict_proba(pdf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf[0].tolist(), [1.0])
        self.assertEqual(pmf[1].tolist(), [1.0])
        self.assertEqual(pmf[2].tolist(), [1.0])
        self.assertEqual(pmf[3].tolist(), [1.0])

    def test_timeout(self):
        with Eventually(180):
            rows = RepairModel() \
                .setTableName("adult") \
                .setRowId("tid") \
                .setErrorCells("adult_dirty") \
                .option("model.hp.max_evals", "10000000") \
                .option("model.hp.no_progress_loss", "100000") \
                .option("model.hp.timeout", "3") \
                .run() \
                .collect()

            self.assertTrue(len(rows), 7)

    def test_training_data_rebalancing(self):
        test_model = self._build_model() \
            .setTableName("mixed_input") \
            .setRowId("tid") \
            .setTrainingDataRebalancingEnabled(True)

        df = test_model.run().orderBy("tid", "attribute")
        self.assertEqual(
            df.selectExpr("tid", "attribute", "current_value").collect(), [
                Row(tid=3, attribute="v3", current_value=None),
                Row(tid=7, attribute="v2", current_value=None),
                Row(tid=10, attribute="v1", current_value=None),
                Row(tid=14, attribute="v4", current_value=None)])

        rows = df.selectExpr("repaired").collect()
        self.assertTrue(rows[0].repaired is not None)
        self.assertTrue(rows[1].repaired is not None)
        self.assertTrue(rows[2].repaired is not None)
        self.assertTrue(rows[3].repaired is not None)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
