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
import tempfile
import unittest
import pandas as pd  # type: ignore[import]

from pyspark import SparkConf
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException

from repair.misc import RepairMisc
from repair.model import FunctionalDepModel, RepairModel, PoorModel
from repair.detectors import ConstraintErrorDetector, NullErrorDetector, RegExErrorDetector
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
            .set("spark.sql.statistics.histogram.numBins", "254") \
            .set("spark.sql.optimizer.excludedRules",
                 "org.apache.spark.sql.catalyst.optimizer.PropagateEmptyRelation")

    @classmethod
    def setUpClass(cls):
        super(RepairModelTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads some test data
        load_testdata(cls.spark, "adult.csv").createOrReplaceTempView("adult")
        load_testdata(cls.spark, "adult_dirty.csv").createOrReplaceTempView("adult_dirty")
        load_testdata(cls.spark, "adult_repair.csv").createOrReplaceTempView("adult_repair")
        load_testdata(cls.spark, "adult_clean.csv").createOrReplaceTempView("adult_clean")

        # Define some expected results
        cls.expected_adult_result_without_repaired = [
            Row(tid="12", attribute="Age", current_value=None),
            Row(tid="12", attribute="Sex", current_value=None),
            Row(tid="16", attribute="Income", current_value=None),
            Row(tid="3", attribute="Sex", current_value=None),
            Row(tid="5", attribute="Age", current_value=None),
            Row(tid="5", attribute="Income", current_value=None),
            Row(tid="7", attribute="Sex", current_value=None)
        ]

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
            "`setRepairDelta` should be called before maximal likelihood repairing",
            lambda: RepairModel().setTableName("dummyTab").setRowId("dummyId")
            .setMaximalLikelihoodRepairEnabled(True).run())
        self.assertRaisesRegexp(
            ValueError,
            "`setRepairDelta` should be called before maximal likelihood repairing",
            lambda: RepairModel().setInput("dummyTab").setRowId("dummyId")
            .setMaximalLikelihoodRepairEnabled(True).run())
        self.assertRaisesRegexp(
            ValueError,
            "`attrs` has at least one attribute",
            lambda: RepairModel().setTargets([]))
        self.assertRaisesRegexp(
            ValueError,
            "threshold must be bigger than 1",
            lambda: RepairModel().setDiscreteThreshold(1))

    def test_exclusive_params(self):
        def _assert_exclusive_params(func):
            self.assertRaisesRegexp(ValueError, "cannot be set to True simultaneously", func)
        test_model = RepairModel()
        api = test_model.setTableName("dummyTab").setRowId("dummyId")
        _assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, compute_repair_candidate_prob=True))
        _assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, compute_training_target_hist=True))
        _assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, repair_data=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_training_target_hist=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, repair_data=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_repair_prob=True))
        _assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_repair_score=True))

    def test_argtype_check(self):
        self.assertRaises(
            TypeError,
            "`db_name` should be provided as str, got int",
            lambda: RepairModel().setDbName(1))
        self.assertRaises(
            TypeError,
            "`table_name` should be provided as str, got int",
            lambda: RepairModel().setTableName(1))
        self.assertRaises(
            TypeError,
            "`thres` should be provided as int, got str",
            lambda: RepairModel().setDiscreteThreshold("a"))
        self.assertRaises(
            TypeError,
            "`thres` should be provided as float, got int",
            lambda: RepairModel().setMinCorrThreshold(1))
        self.assertRaises(
            TypeError,
            "`beta` should be provided as float, got int",
            lambda: RepairModel().setDomainThresholds(1.0, 1))
        self.assertRaises(
            TypeError,
            "`input` should be provided as str/DataFrame, got int",
            lambda: RepairModel().setInput(1))
        self.assertRaises(
            TypeError,
            "`attrs` should be provided as list[str], got int",
            lambda: RepairModel().setTargets(1))
        self.assertRaises(
            TypeError,
            "`attrs` should be provided as list[str], got int in elements",
            lambda: RepairModel().setTargets(["a", 1]))
        self.assertRaises(
            TypeError,
            "`detectors` should be provided as list[ErrorDetector], got int in elements",
            lambda: RepairModel().setErrorDetectors([1]))
        self.assertRaises(
            TypeError,
            "`cf` should be provided as UpdateCostFunction, got int",
            lambda: RepairModel().setUpdateCostFunction([1]))

    # TODO: We fix a seed for building a repair model, but inferred values fluctuate run-by-run.
    # So, to avoid it, we set 1 to `hp.max_evals` for now.
    def _build_model(self):
        return RepairModel().option("hp.max_evals", "1")

    def test_multiple_run(self):
        # Checks if auto-generated views are dropped finally
        current_view_nums = self.spark.sql("SHOW VIEWS").count()

        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        def _test_basic():
            test_model = self._build_model() \
                .setTableName("adult") \
                .setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

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
            df.selectExpr("tid", "attribute", "current_value").orderBy("tid", "attribute").collect(),
            self.expected_adult_result_without_repaired)

    def test_table_input(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        with self.table("adult_table"):

            # Tests for `setDbName`
            self.spark.table("adult").write.mode("overwrite").saveAsTable("adult_table")
            test_model = self._build_model() \
                .setDbName("default") \
                .setTableName("adult_table") \
                .setRowId("tid")

            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

    def test_input_overwrite(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

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
                expected_result)

    def test_setInput(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        def _test_setInput(input):
            test_model = self._build_model().setInput(input).setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

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
                .collect()
            self.assertEqual(
                actual_result.collect(),
                expected_result)

        _test_setTargets(["Sex"])
        _test_setTargets(["Sex", "Income"])
        _test_setTargets(["Age", "Sex"])
        _test_setTargets(["Non-Existent", "Age"])
        _test_setTargets(["Non-Existent"])

    def test_setErrorCells(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        def _test_setErrorCells(error_cells):
            test_model = self._build_model() \
                .setTableName("adult") \
                .setRowId("tid") \
                .setErrorCells(error_cells)
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

        _test_setErrorCells("adult_dirty")
        _test_setErrorCells(self.spark.table("adult_dirty"))

    def _assert_adult_without_repaired(self, test_model):
        def _test(df):
            self.assertEqual(
                df.selectExpr("tid", "attribute", "current_value").orderBy("tid", "attribute").collect(),
                self.expected_adult_result_without_repaired)
        _test(test_model.setParallelStatTrainingEnabled(False).run())
        _test(test_model.setParallelStatTrainingEnabled(True).run())

    def test_setMaxTrainingRowNum(self):
        row_num = int(self.spark.table("adult").count() / 2)
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setMaxTrainingRowNum(row_num)
        self._assert_adult_without_repaired(test_model)

    def test_setMaxTrainingColumnNum(self):
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setMaxTrainingColumnNum(2)
        self._assert_adult_without_repaired(test_model)

    def test_error_cells_having_no_existent_attribute(self):
        error_cells = [
            Row(tid="1", attribute="NoExistent"),
            Row(tid="5", attribute="Income"),
            Row(tid="16", attribute="Income")
        ]
        error_cells_df = self.spark.createDataFrame(
            error_cells, schema="tid STRING, attribute STRING")
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .setErrorCells(error_cells_df)
        self.assertEqual(
            test_model.run().orderBy("tid", "attribute").collect(), [
                Row(tid="16", attribute="Income", current_value=None, repaired="MoreThan50K"),
                Row(tid="5", attribute="Income", current_value=None, repaired="MoreThan50K")])

    def test_detect_errors_only(self):
        # Tests for `NullErrorDetector`
        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(),
            self.expected_adult_result_without_repaired)

        # Tests for `RegExErrorDetector`
        error_detectors = [
            NullErrorDetector(),
            RegExErrorDetector("Exec-managerial"),
            RegExErrorDetector("India")
        ]
        regex_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            regex_errors.subtract(null_errors).orderBy("tid", "attribute").collect(), [
                Row(tid="1", attribute="Occupation", current_value="Exec-managerial"),
                Row(tid="12", attribute="Occupation", current_value="Exec-managerial"),
                Row(tid="14", attribute="Occupation", current_value="Exec-managerial"),
                Row(tid="16", attribute="Occupation", current_value="Exec-managerial"),
                Row(tid="7", attribute="Country", current_value="India")])

        # Tests for `ConstraintErrorDetector`
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        error_detectors = [
            NullErrorDetector(),
            ConstraintErrorDetector(constraint_path)
        ]
        constraint_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setErrorDetectors(error_detectors) \
            .run(detect_errors_only=True)
        self.assertEqual(
            constraint_errors.subtract(null_errors).orderBy("tid", "attribute").collect(), [
                Row(tid="11", attribute="Relationship", current_value="Husband"),
                Row(tid="11", attribute="Sex", current_value="Female"),
                Row(tid="4", attribute="Relationship", current_value="Husband"),
                Row(tid="4", attribute="Sex", current_value="Female")])

    def test_repair_data(self):
        test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid")
        expected_result = self.spark.table("adult_clean") \
            .orderBy("tid").collect()
        self.assertEqual(
            test_model.run(repair_data=True).orderBy("tid").collect(),
            expected_result)

    def test_unsupported_types(self):
        with self.tempView("inputView"):
            self.spark.range(1).selectExpr("id tid", "1 AS x", "CAST('2021-08-01' AS DATE) y") \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
                .setRowId("tid")
            self.assertRaises(AnalysisException, lambda: test_model.run())

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
            self.assertRaises(AnalysisException, lambda: test_model.run())

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
            self.assertRaises(AnalysisException, lambda: test_model.run())

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

    def test_no_valid_noisy_cell_exists(self):
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
                "Noisy cells have valid discrete properties for domain analysis",
                lambda: test_model.run())

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

    def test_rule_based_model(self):
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
                    .setRuleBasedModelEnabled(True)

                self.assertEqual(
                    test_model.run().orderBy("tid", "attribute").collect(), [
                        Row(tid=3, attribute="y", current_value=None, repaired="test-1"),
                        Row(tid=5, attribute="y", current_value=None, repaired="test-2"),
                        Row(tid=6, attribute="y", current_value=None, repaired=None)])

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

    def _check_repair_prob_and_score(self, df, expected_schema):
        self.assertEqual(
            df.schema.simpleString(),
            expected_schema)

        expected_result = self.spark.table("adult_repair") \
            .selectExpr("tid", "attribute") \
            .orderBy("tid")\
            .collect()
        self.assertEqual(
            df.selectExpr("tid", "attribute").orderBy("tid").collect(),
            expected_result)

    def test_compute_repair_candidate_prob(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .run(compute_repair_candidate_prob=True)

        self._check_repair_prob_and_score(
            repaired_df,
            "struct<tid:string,attribute:string,current:struct<value:string,prob:double>,"
            "pmf:array<struct<c:string,p:double>>>")

    def test_compute_repair_prob(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .run(compute_repair_prob=True)

        self._check_repair_prob_and_score(
            repaired_df,
            "struct<tid:string,attribute:string,current_value:string,"
            "repaired:string,prob:double>")

    def test_compute_repair_prob_and_score(self):
        repaired_df = test_model = self._build_model() \
            .setTableName("adult") \
            .setRowId("tid") \
            .run(compute_repair_score=True)

        self._check_repair_prob_and_score(
            repaired_df,
            "struct<tid:string,attribute:string,current_value:string,"
            "repaired:string,score:double>")

    def test_rule_based_model(self):
        model = FunctionalDepModel("x", {1: "test-1", 2: "test-2", 3: "test-3"})
        pdf = pd.DataFrame([[3], [1], [2], [4]], columns=["x"])
        self.assertEqual(model.classes_.tolist(), ["test-1", "test-2", "test-3"])
        self.assertEqual(model.predict(pdf), ["test-3", "test-1", "test-2", None])
        pmf = model.predict_proba(pdf)
        self.assertEqual(len(pmf), 4)
        self.assertEqual(pmf[0].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(pmf[1].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(pmf[2].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(pmf[3].tolist(), [0.0, 0.0, 0.0])

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
                .option("hp.max_evals", "10000000") \
                .option("hp.no_progress_loss", "100000") \
                .option("hp.timeout", "3") \
                .run() \
                .collect()

            self.assertTrue(len(rows), 7)

    def test_training_data_rebalancing(self):
        with self.tempView("inputView"):
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
            self.spark.createDataFrame(rows, ["tid", "v1", "v2", "v3", "v4"]) \
                .createOrReplaceTempView("inputView")
            test_model = self._build_model() \
                .setTableName("inputView") \
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
