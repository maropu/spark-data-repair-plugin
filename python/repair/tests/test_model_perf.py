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

from repair.model import RepairModel
from repair.detectors import ConstraintErrorDetector
from repair.tests.requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message
from repair.tests.testutils import ReusedSQLTestCase, load_testdata


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)  # type: ignore
class RepairModelPerformanceTests(ReusedSQLTestCase):

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
        super(RepairModelPerformanceTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads test data
        load_testdata(cls.spark, "iris_clean.csv").createOrReplaceTempView("iris_clean")
        load_testdata(cls.spark, "iris_test1.csv").createOrReplaceTempView("iris_test1")
        load_testdata(cls.spark, "iris_test2.csv").createOrReplaceTempView("iris_test2")

        boston_schema = "tid string, CRIM double, ZN double, INDUS double, CHAS double, " \
            "NOX double, RM double, AGE double, DIS double, RAD double, TAX double, " \
            "PTRATIO double, B double, LSTAT double"
        load_testdata(cls.spark, "boston_clean.csv").createOrReplaceTempView("boston_clean")
        load_testdata(cls.spark, "boston_test1.csv", boston_schema) \
            .createOrReplaceTempView("boston_test1")
        load_testdata(cls.spark, "boston_test2.csv", boston_schema) \
            .createOrReplaceTempView("boston_test2")

        load_testdata(cls.spark, "hospital_clean.csv").createOrReplaceTempView("hospital_clean")
        load_testdata(cls.spark, "hospital.csv").createOrReplaceTempView("hospital")
        load_testdata(cls.spark, "hospital_error_cells.csv") \
            .createOrReplaceTempView("hospital_error_cells")

    @classmethod
    def tearDownClass(cls):
        super(ReusedSQLTestCase, cls).tearDownClass()

    def _compute_rmse(self, repaired_df, expected):
        # Compares predicted values with the correct ones
        cmp_df = repaired_df.join(self.spark.table(expected), ["tid", "attribute"], "inner")
        n = repaired_df.count()
        return cmp_df.selectExpr(f"sqrt(sum(pow(correct_val - repaired, 2.0)) / {n}) rmse") \
            .collect()[0] \
            .rmse

    def test_perf_iris_target_num_1(self):
        # Target column is "sepal_width"
        repaired_df = RepairModel() \
            .setTableName("iris_test1") \
            .setRowId("tid") \
            .run()
        rmse = self._compute_rmse(repaired_df, "iris_clean")
        self.assertLess(rmse, 0.48, msg=f"target:iris(sepal_width)")

    def test_perf_iris_target_num_2(self):
        # Target columns are "sepal_length" and "petal_width"
        repaired_df = RepairModel() \
            .setTableName("iris_test2") \
            .setRowId("tid") \
            .run()
        rmse = self._compute_rmse(repaired_df, "iris_clean")
        self.assertLess(rmse, 0.36, msg=f"target:iris(sepal_length,petal_width)")

    def test_perf_boston_target_num_1(self):
        # Target column is "AGE"
        repaired_df = RepairModel() \
            .setTableName("boston_test2") \
            .setRowId("tid") \
            .run()
        rmse = self._compute_rmse(repaired_df, "boston_clean")
        # TODO: Needs to tune the performance below
        self.assertLess(rmse, 11.5, msg=f"target:boston(AGE)")

    def test_perf_boston_target_num_3(self):
        # Target columns are "CRIM", "LSTAT", and "RM"
        repaired_df = RepairModel() \
            .setTableName("boston_test1") \
            .setRowId("tid") \
            .run()
        # TODO: Needs to tune the performance below
        rmse = self._compute_rmse(repaired_df, "boston_clean")
        self.assertLess(rmse, 3.20, msg=f"target:boston(CRIM,LSTAT,RM)")

    @unittest.skip(reason="much time to compute repaired data")
    def test_perf_hospital(self):
        constraint_path = "{}/hospital_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        repaired_df = RepairModel() \
            .setTableName("hospital") \
            .setRowId("tid") \
            .setErrorDetector(ConstraintErrorDetector(constraint_path)) \
            .setDiscreteThreshold(100) \
            .run()

        # All the values of "Score" column is NULL, so ignores it
        pdf = repaired_df.join(
            self.spark.table("hospital_clean").where("attribute != 'Score'"),
            ["tid", "attribute"], "inner")
        rdf = repaired_df.join(
            self.spark.table("hospital_error_cells").where("attribute != 'Score'"),
            ["tid", "attribute"], "right_outer")

        # Computes performance numbers (precision & recall)
        #  - Precision: the fraction of correct repairs, i.e., repairs that match
        #    the ground truth, over the total number of repairs performed
        #  - Recall: correct repairs over the total number of errors
        precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
        recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
        f1 = (2.0 * precision * recall) / (precision + recall)

        self.assertTrue(
            precision > 0.70 and recall > 0.65 and f1 > 0.67,
            msg=f"target:hospital precision:{precision} recall:{recall} f1:{f1}")


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
