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

from testutils import ReusedSQLTestCase
from requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message

from pyspark import SparkConf

from repair.model import RepairModel


def load_testdata(spark, filename):
    fmt = os.path.splitext(filename)[1][1:]
    return spark.read.format(fmt) \
        .option("header", True) \
        .load("{}/{}".format(os.getenv("REPAIR_TESTDATA"), filename))


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
            "`setRepairDelta` should be called before maximal likelihood repairing",
            lambda: RepairModel().setTableName("dummyTab").setRowId("dummyId")
            .setMaximalLikelihoodRepairEnabled(True).run())
        self.assertRaisesRegexp(
            ValueError,
            "Inference order must be `error`, `domain`, or `entropy`",
            lambda: RepairModel().setTableName("dummyTab").setRowId("dummyId")
            .setInferenceOrder("invalid").run())

    def __assert_exclusive_params(self, func):
        self.assertRaisesRegexp(ValueError, "cannot be set to True simultaneously", func)

    def test_exclusive_params(self):
        test_model = RepairModel()
        api = test_model.setTableName("dummyTab").setRowId("dummyId")
        self.__assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, compute_repair_candidate_prob=True))
        self.__assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, compute_training_target_hist=True))
        self.__assert_exclusive_params(
            lambda: api.run(detect_errors_only=True, repair_data=True))
        self.__assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, compute_training_target_hist=True))
        self.__assert_exclusive_params(
            lambda: api.run(compute_repair_candidate_prob=True, repair_data=True))

    def test_basic(self):
        test_model = RepairModel()
        df = test_model.setTableName("adult").setRowId("tid").setErrorCells("adult_dirty").run()
        repair_expected_df = load_testdata(self.spark, "adult_repair_expected.csv")
        self.assertEqual(
            df.orderBy("tid", "attribute").collect(),
            repair_expected_df.orderBy("tid", "attribute").collect())

    def test_version(self):
        self.assertEqual(RepairModel().version(), "0.1.0-spark3.0-EXPERIMENTAL")


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
