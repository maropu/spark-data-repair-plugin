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
from pyspark.sql.utils import AnalysisException

from repair.detectors import ConstraintErrorDetector, NullErrorDetector, \
    OutlierErrorDetector, RegExErrorDetector
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

    def test_invalid_targets(self):
        self.assertRaisesRegexp(
            AnalysisException,
            "Target attributes not found in adult: Unknown",
            lambda: NullErrorDetector().setUp("tid", "adult", ["Unknown"]).detect())
        self.assertRaisesRegexp(
            AnalysisException,
            "Target attributes not found in adult: Unknown",
            lambda: RegExErrorDetector("Exec-managerial").setUp("tid", "adult", ["Unknown"]).detect())
        self.assertRaisesRegexp(
            AnalysisException,
            "Target attributes not found in adult: Unknown",
            lambda: ConstraintErrorDetector("notused").setUp("tid", "adult", ["Unknown"]).detect())
        self.assertRaisesRegexp(
            AnalysisException,
            "Target attributes not found in adult: Unknown",
            lambda: OutlierErrorDetector().setUp("tid", "adult", ["Unknown"]).detect())

    def test_NullErrorDetector(self):
        errors = NullErrorDetector().setUp("tid", "adult", []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex"),
                Row(tid=5, attribute="Age"),
                Row(tid=5, attribute="Income"),
                Row(tid=7, attribute="Sex"),
                Row(tid=12, attribute="Age"),
                Row(tid=12, attribute="Sex"),
                Row(tid=16, attribute="Income")])
        errors = NullErrorDetector().setUp("tid", "adult", ["Sex"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=3, attribute="Sex"),
                Row(tid=7, attribute="Sex"),
                Row(tid=12, attribute="Sex")])
        errors = NullErrorDetector().setUp("tid", "adult", ["Age", "Income"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Age"),
                Row(tid=5, attribute="Income"),
                Row(tid=12, attribute="Age"),
                Row(tid=16, attribute="Income")])
        errors = NullErrorDetector().setUp("tid", "adult", ["Income", "Unknown"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=5, attribute="Income"),
                Row(tid=16, attribute="Income")])

    def test_RegExErrorDetector(self):
        errors = RegExErrorDetector("Exec-managerial", error_cells_as_string=False) \
            .setUp("tid", "adult", []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=1, attribute="Occupation"),
                Row(tid=12, attribute="Occupation"),
                Row(tid=14, attribute="Occupation"),
                Row(tid=16, attribute="Occupation")])
        errors = RegExErrorDetector("Exec-managerial", error_cells_as_string=False) \
            .setUp("tid", "adult", ["Occupation"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=1, attribute="Occupation"),
                Row(tid=12, attribute="Occupation"),
                Row(tid=14, attribute="Occupation"),
                Row(tid=16, attribute="Occupation")])
        errors = RegExErrorDetector("Exec-managerial", error_cells_as_string=False) \
            .setUp("tid", "adult", ["Unknown", "Occupation"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=1, attribute="Occupation"),
                Row(tid=12, attribute="Occupation"),
                Row(tid=14, attribute="Occupation"),
                Row(tid=16, attribute="Occupation")])

        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, 12), (2, 123), (3, 1234), (4, 12345)], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            for (error_cells_as_string, expected) in \
                    ((False, []), (True, [Row(tid=3, attribute="v"), Row(tid=4, attribute="v")])):
                errors = RegExErrorDetector("123.+", error_cells_as_string) \
                    .setUp("tid", "tempView", []).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    expected)

    def test_ConstraintErrorDetector(self):
        constraint_path = "{}/adult_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", []).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Relationship"),
                Row(tid=11, attribute="Sex")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", ["Relationship"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=11, attribute="Relationship")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", ["Sex", "Relationship"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Relationship"),
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Relationship"),
                Row(tid=11, attribute="Sex")])
        errors = ConstraintErrorDetector(constraint_path) \
            .setUp("tid", "adult", ["Unknown", "Sex"]).detect()
        self.assertEqual(
            errors.orderBy("tid", "attribute").collect(), [
                Row(tid=4, attribute="Sex"),
                Row(tid=11, attribute="Sex")])

    def test_OutlierErrorDetector(self):
        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, 1.0), (2, 1.0), (3, 1.0), (4, 1000.0)], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            for approx_enabled in [True, False]:
                errors = OutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", []).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])
                errors = OutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["v"]).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])
                errors = OutlierErrorDetector(approx_enabled) \
                    .setUp("tid", "tempView", ["Unknown", "v"]).detect()
                self.assertEqual(
                    errors.orderBy("tid", "attribute").collect(),
                    [Row(tid=4, attribute="v")])


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
