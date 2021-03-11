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
import glob
import tempfile
import unittest
import pandas as pd  # type: ignore[import]

from pyspark import SparkConf
from pyspark.sql import Row

from repair.misc import RepairMisc
from repair.model import FunctionalDepModel, RepairModel
from repair.detectors import ConstraintErrorDetector, RegExErrorDetector
from repair.tests.requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message
from repair.tests.testutils import ReusedSQLTestCase, load_testdata


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)  # type: ignore
class RepairModelTests(ReusedSQLTestCase):

    @classmethod
    def conf(cls):
        return SparkConf() \
            .set("spark.jars", os.getenv("REPAIR_API_LIB")) \
            .set("spark.sql.crossJoin.enabled", "true") \
            .set("spark.sql.cbo.enabled", "true") \
            .set("spark.sql.statistics.histogram.enabled", "true") \
            .set("spark.sql.statistics.histogram.numBins", "254")

    @classmethod
    def setUpClass(cls):
        super(RepairModelTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads test data
        load_testdata(cls.spark, "adult.csv").createOrReplaceTempView("adult")
        load_testdata(cls.spark, "adult_dirty.csv").createOrReplaceTempView("adult_dirty")
        load_testdata(cls.spark, "adult_repair.csv").createOrReplaceTempView("adult_repair")
        load_testdata(cls.spark, "adult_clean.csv").createOrReplaceTempView("adult_clean")

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
            "Inference order must be `error`, `domain`, or `entropy`",
            lambda: RepairModel().setTableName("dummyTab").setRowId("dummyId")
            .setInferenceOrder("invalid").run())
        self.assertRaisesRegexp(
            ValueError,
            "Inference order must be `error`, `domain`, or `entropy`",
            lambda: RepairModel().setInput("dummyTab").setRowId("dummyId")
            .setInferenceOrder("invalid").run())
        self.assertRaisesRegexp(
            ValueError,
            "`attrs` has at least one attribute",
            lambda: RepairModel().setTargets([]))

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

    def test_setCheckpointPath(self):
        with tempfile.TemporaryDirectory() as path:
            checkpoint_path = f"{path}/chkpnt"
            self._build_model() \
                .setTableName("adult") \
                .setRowId("tid") \
                .setCheckpointPath(checkpoint_path) \
                .run()

            files = glob.glob(f"{checkpoint_path}/*")
            expected_files = [
                "0_classifier_Income.json",
                "0_classifier_Income.pkl",
                "1_classifier_Sex.json",
                "1_classifier_Sex.pkl",
                "2_classifier_Age.json",
                "2_classifier_Age.pkl",
                "metadata.json"
            ]
            for file in files:
                fn = os.path.basename(file)
                self.assertTrue(fn in expected_files)

            self.assertRaisesRegexp(
                ValueError,
                f"Path '{checkpoint_path}' already exists",
                lambda: RepairModel().setCheckpointPath(checkpoint_path))

    def test_detect_errors_only(self):
        # Tests for `NullErrorDetector`
        null_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .run(detect_errors_only=True)
        self.assertEqual(
            null_errors.orderBy("tid", "attribute").collect(), [
                Row(tid="12", attribute="Age", current_value=None),
                Row(tid="12", attribute="Sex", current_value=None),
                Row(tid="16", attribute="Income", current_value=None),
                Row(tid="3", attribute="Sex", current_value=None),
                Row(tid="5", attribute="Age", current_value=None),
                Row(tid="5", attribute="Income", current_value=None),
                Row(tid="7", attribute="Sex", current_value=None)])

        # Tests for `RegExErrorDetector`
        regex_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setErrorDetector(RegExErrorDetector("Exec-managerial")) \
            .setErrorDetector(RegExErrorDetector("India")) \
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
        constraint_errors = self._build_model() \
            .setInput("adult") \
            .setRowId("tid") \
            .setErrorDetector(ConstraintErrorDetector(constraint_path)) \
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

    def test_setRuleBasedModelEnabled(self):
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

                test_model = self._build_model() \
                    .setTableName("inputView") \
                    .setRowId("tid") \
                    .setErrorCells("errorCells") \
                    .setErrorDetector(ConstraintErrorDetector(f.name)) \
                    .setRuleBasedModelEnabled(True)

                self.assertEqual(
                    test_model.run().orderBy("tid", "attribute").collect(), [
                        Row(tid=3, attribute="y", current_value=None, repaired="test-1"),
                        Row(tid=5, attribute="y", current_value=None, repaired="test-2"),
                        Row(tid=6, attribute="y", current_value=None, repaired=None)])

    def test_setRepairUpdates(self):
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

    def test_FunctionalDepModel(self):
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


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
