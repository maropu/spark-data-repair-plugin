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
import unittest

from testutils import ReusedSQLTestCase, load_testdata
from requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message

from pyspark import SparkConf
from pyspark.sql import Row

from repair.model import RepairModel
from repair.detectors import ConstraintErrorDetector, RegExErrorDetector


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)
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

    def test_multiple_run(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        def _test_basic():
            test_model = RepairModel() \
                .setTableName("adult") \
                .setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

        _test_basic()  # first run
        _test_basic()  # second run

    # TODO: Fix a test failure in the test below:
    @unittest.skip(reason="")
    def test_table_input(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        with self.table("adult_table"):
            # Tests for `setDbName`
            self.spark.table("adult").write.mode("overwrite").saveAsTable("adult_table")
            test_model = RepairModel() \
                .setDbName("default") \
                .setTableName("adult_table") \
                .setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

    # TODO: Fix a test failure in the test below:
    @unittest.skip(reason="")
    def test_input_overwrite(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        with self.table("adult_table"):
            # Tests for input overwrite case ("default.adult_table" -> "adult")
            self.spark.table("adult").write.mode("overwrite").saveAsTable("adult_table")
            test_model = RepairModel() \
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
            test_model = RepairModel().setInput(input).setRowId("tid")
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

        _test_setInput("adult")
        # _test_setInput(self.spark.table("adult"))

    def test_setErrorCells(self):
        expected_result = self.spark.table("adult_repair") \
            .orderBy("tid", "attribute").collect()

        def _test_setErrorCells(error_cells):
            test_model = RepairModel() \
                .setTableName("adult") \
                .setRowId("tid") \
                .setErrorCells(error_cells)
            self.assertEqual(
                test_model.run().orderBy("tid", "attribute").collect(),
                expected_result)

        _test_setErrorCells("adult_dirty")
        _test_setErrorCells(self.spark.table("adult_dirty"))

    def test_detect_errors_only(self):
        # Tests for `NullErrorDetector`
        null_errors = RepairModel() \
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
        regex_errors = RepairModel() \
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
        constraint_errors = RepairModel() \
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

    def test_version(self):
        self.assertEqual(RepairModel().version(), "0.1.0-spark3.0-EXPERIMENTAL")


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
