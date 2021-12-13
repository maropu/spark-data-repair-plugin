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

from pyspark import SparkConf
from pyspark.sql import Row

from repair.detectors import ConstraintErrorDetector, DomainValues, NullErrorDetector, \
    LOFOutlierErrorDetector, GaussianOutlierErrorDetector, ScikitLearnBackedErrorDetector, RegExErrorDetector
from repair.tests.testutils import ReusedSQLTestCase, load_testdata


class ErrorDetectorTests(ReusedSQLTestCase):

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
        super(ErrorDetectorTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads test data
        load_testdata(cls.spark, "adult.csv").createOrReplaceTempView("adult")

    def test_NullErrorDetector(self):
        errors = NullErrorDetector().setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex"),
                Row(tid=5, attribute="Age"),
                Row(tid=5, attribute="Income"),
                Row(tid=7, attribute="Sex"),
                Row(tid=12, attribute="Age"),
                Row(tid=12, attribute="Sex"),
                Row(tid=16, attribute="Income")])
        errors = NullErrorDetector().setUp("tid", "adult", [], ["Sex"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex"),
                Row(tid=7, attribute="Sex"),
                Row(tid=12, attribute="Sex")])
        errors = NullErrorDetector().setUp("tid", "adult", [], ["Age", "Income"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Age"),
                Row(tid=5, attribute="Income"),
                Row(tid=12, attribute="Age"),
                Row(tid=16, attribute="Income")])
        errors = NullErrorDetector().setUp("tid", "adult", [], ["Income", "Unknown"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income"),
                Row(tid=16, attribute="Income")])

    def test_NullErrorDetector_empty_result(self):
        errors = NullErrorDetector().setUp("tid", "adult", [], ["Non-existent"]).detect()
        self.assertEqual(errors.collect(), [])

    def test_DomainValues(self):
        errors = DomainValues("Country", []) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(),
            [Row(tid=i, attribute="Country") for i in range(0, 20)])
        errors = DomainValues("Country", ["United-States"]) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country"),
                Row(tid=19, attribute="Country")])
        errors = DomainValues("Income", ["LessThan50K", "MoreThan50K"]) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income"),
                Row(tid=16, attribute="Income")])

    def test_DomainValues_autofill(self):
        errors = DomainValues("Country", autofill=True, min_count_thres=4) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country"),
                Row(tid=19, attribute="Country")])
        errors = DomainValues("Income", autofill=True, min_count_thres=1) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income"),
                Row(tid=16, attribute="Income")])

    def test_DomainValues_empty_result(self):
        errors = DomainValues("Country", []) \
            .setUp("tid", "adult", [], ['Non-existent']).detect()
        self.assertEqual(errors.collect(), [])

    def test_RegExErrorDetector(self):
        errors = RegExErrorDetector("Country", "United-States") \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country"),
                Row(tid=19, attribute="Country")])
        errors = RegExErrorDetector("Country", "United-States") \
            .setUp("tid", "adult", [], ["Country"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country"),
                Row(tid=19, attribute="Country")])
        errors = RegExErrorDetector("Country", "United-States") \
            .setUp("tid", "adult", [], ["Unknown", "Country"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=7, attribute="Country"),
                Row(tid=19, attribute="Country")])

        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, 12), (2, 123), (3, 1234), (4, 12345)], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            errors = RegExErrorDetector("v", "123.+").setUp("tid", "tempView", [], []).detect()
            self.assertEqual(
                errors.orderBy("tid", "attribute").collect(),
                [Row(tid=1, attribute="v"), Row(tid=2, attribute="v")])

    def test_RegExErrorDetector_empty_result(self):
        errors = RegExErrorDetector("Country", "United-States") \
            .setUp("tid", "adult", [], ['Non-existent']).detect()
        self.assertEqual(errors.collect(), [])

    def test_ConstraintErrorDetector(self):
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Relationship"),
                Row(tid=11, attribute="Sex")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], ["Relationship"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=11, attribute="Relationship")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], ["Sex", "Relationship"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Relationship"),
                Row(tid=11, attribute="Sex")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], ["Unknown", "Sex"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Sex")])

    def test_ConstraintErrorDetector_empty_result(self):
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], ['Non-existent']).detect()
        self.assertEqual(errors.collect(), [])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", [], ['Income']).detect()
        self.assertEqual(errors.collect(), [])

    def test_GaussianOutlierErrorDetector(self):
        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, 1.0), (2, 1.0), (3, 1.0), (4, 1000.0), (5, None)], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            for approx_enabled in [True, False]:
                errors = GaussianOutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["v"], []).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])
                errors = GaussianOutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["v"], ["v"]).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])
                errors = GaussianOutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["v"], ["Unknown", "v"]).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])

    def test_GaussianOutlierErrorDetector_empty_result(self):
        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, 1.0), (2, 1.0), (3, 1.0), (4, 1000.0), (5, None)], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            for approx_enabled in [True, False]:
                errors = GaussianOutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["v"], ["Non-existent"]).detect()
                self.assertEqual(errors.collect(), [])

    def test_LOFOutlierErrorDetector(self):
        def _test(input_nrows: int, parallel_mode_threshold: int):
            with self.tempView("tempView"):
                normal_df = self.spark.range(input_nrows).selectExpr('id', 'id % 2 v1', 'id % 3 v2')
                dirty_data = [(1000000, 1, 1000), (1000001, 1000, 1), (1000002, None, None)]
                dirty_df = self.spark.createDataFrame(dirty_data, ["id", "v1", "v2"])
                normal_df.union(dirty_df).createOrReplaceTempView("tempView")

                self.assertRaisesRegexp(
                    ValueError,
                    "`num_parallelism` must be positive, got 0",
                    lambda: LOFOutlierErrorDetector(parallel_mode_threshold, num_parallelism=0))

                errors = LOFOutlierErrorDetector(parallel_mode_threshold, num_parallelism=1) \
                    .setUp("id", "tempView", ["v1", "v2"], []).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000000, attribute='v2'), Row(id=1000001, attribute='v1')])
                errors = LOFOutlierErrorDetector(parallel_mode_threshold, num_parallelism=1) \
                    .setUp("id", "tempView", ["v1", "v2"], ["v1"]).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000001, attribute='v1')])
                errors = LOFOutlierErrorDetector(parallel_mode_threshold, num_parallelism=1) \
                    .setUp("id", "tempView", ["v1", "v2"], ["Unknown", "v1"]).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000001, attribute='v1')])

                errors = LOFOutlierErrorDetector(parallel_mode_threshold, num_parallelism=1) \
                    .setUp("id", "tempView", ["v1", "v2"], ["Non-existent"]).detect()
                self.assertEqual(errors.collect(), [])

        _test(input_nrows=3000, parallel_mode_threshold=5000)
        _test(input_nrows=10000, parallel_mode_threshold=5000)

    def test_ScikitLearnBackedErrorDetector(self):
        from sklearn.neighbors import LocalOutlierFactor

        def _test(input_nrows: int, parallel_mode_threshold: int):
            with self.tempView("tempView"):
                normal_df = self.spark.range(input_nrows).selectExpr('id', 'id % 2 v1', 'id % 3 v2')
                dirty_data = [(1000000, 1, 1000), (1000001, 1000, 1), (1000002, None, None)]
                dirty_df = self.spark.createDataFrame(dirty_data, ["id", "v1", "v2"])
                normal_df.union(dirty_df).createOrReplaceTempView("tempView")

                error_detector_cls = lambda: LocalOutlierFactor(novelty=False)

                bad_params = {
                    'error_detector_cls': 1,
                    'parallel_mode_threshold': parallel_mode_threshold,
                    'num_parallelism': 1
                }
                self.assertRaisesRegexp(
                    ValueError,
                    "`error_detector_cls` should be callable",
                    lambda: ScikitLearnBackedErrorDetector(**bad_params))

                bad_params = {
                    'error_detector_cls': lambda: 1,
                    'parallel_mode_threshold': parallel_mode_threshold,
                    'num_parallelism': 1
                }
                self.assertRaisesRegexp(
                    ValueError,
                    "An instance that `error_detector_cls` returns should have a `fit_predict` method",
                    lambda: ScikitLearnBackedErrorDetector(**bad_params))

                bad_params = {
                    'error_detector_cls': error_detector_cls,
                    'parallel_mode_threshold': parallel_mode_threshold,
                    'num_parallelism': 0
                }
                self.assertRaisesRegexp(
                    ValueError,
                    "`num_parallelism` must be positive, got 0",
                    lambda: ScikitLearnBackedErrorDetector(**bad_params))

                params = {
                    'error_detector_cls': error_detector_cls,
                    'parallel_mode_threshold': parallel_mode_threshold,
                    'num_parallelism': 1
                }
                errors = ScikitLearnBackedErrorDetector(**params) \
                    .setUp("id", "tempView", ["v1", "v2"], []).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000000, attribute='v2'), Row(id=1000001, attribute='v1')])
                errors = ScikitLearnBackedErrorDetector(**params) \
                    .setUp("id", "tempView", ["v1", "v2"], ["v1"]).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000001, attribute='v1')])
                errors = ScikitLearnBackedErrorDetector(**params) \
                    .setUp("id", "tempView", ["v1", "v2"], ["Unknown", "v1"]).detect()
                self.assertEqual(
                    errors.orderBy("id", "attribute").collect(),
                    [Row(id=1000001, attribute='v1')])

                errors = ScikitLearnBackedErrorDetector(**params) \
                    .setUp("id", "tempView", ["v1", "v2"], ["Non-existent"]).detect()
                self.assertEqual(errors.collect(), [])

        _test(input_nrows=3000, parallel_mode_threshold=5000)
        _test(input_nrows=10000, parallel_mode_threshold=5000)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
