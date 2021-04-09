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
import logging
import unittest

from pyspark import SparkConf

from repair.misc import RepairMisc
from repair.model import RepairModel
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
            .set("spark.sql.cbo.enabled", "true") \
            .set("spark.sql.statistics.histogram.enabled", "true") \
            .set("spark.sql.statistics.histogram.numBins", "254") \
            .set("spark.sql.optimizer.excludedRules",
                 "org.apache.spark.sql.catalyst.optimizer.PropagateEmptyRelation")

    @classmethod
    def setUpClass(cls):
        super(RepairModelPerformanceTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads test data
        load_testdata(cls.spark, "iris_clean.csv").createOrReplaceTempView("iris_clean")
        load_testdata(cls.spark, "iris_orig.csv").createOrReplaceTempView("iris")

        boston_schema = "tid string, CRIM double, ZN string, INDUS double, CHAS string, " \
            "NOX double, RM double, AGE double, DIS double, RAD string, TAX double, " \
            "PTRATIO double, B double, LSTAT double"
        load_testdata(cls.spark, "boston_clean.csv").createOrReplaceTempView("boston_clean")
        load_testdata(cls.spark, "boston_orig.csv", boston_schema) \
            .createOrReplaceTempView("boston")

        load_testdata(cls.spark, "hospital_clean.csv").createOrReplaceTempView("hospital_clean")
        load_testdata(cls.spark, "hospital.csv").createOrReplaceTempView("hospital")
        load_testdata(cls.spark, "hospital_error_cells.csv") \
            .createOrReplaceTempView("hospital_error_cells")

    @classmethod
    def tearDownClass(cls):
        super(ReusedSQLTestCase, cls).tearDownClass()

    def _inject_null(self, table_name, target_attr_list):
        return RepairMisc() \
            .option("table_name", table_name) \
            .option("target_attr_list", target_attr_list) \
            .option("null_ratio", "0.10") \
            .option("seed", "0") \
            .injectNull()

    def _build_model(self, input):
        return RepairModel() \
            .setInput(input) \
            .setRowId("tid") \
            .option("hp.no_progress_loss", "150")

    def _compute_rmse(self, repaired_df, expected):
        # Compares predicted values with the correct ones
        cmp_df = repaired_df.join(self.spark.table(expected), ["tid", "attribute"], "inner")
        n = repaired_df.count()
        return cmp_df.selectExpr(f"sqrt(sum(pow(correct_val - repaired, 2.0)) / {n}) rmse") \
            .collect()[0] \
            .rmse

    @unittest.skip(reason="")
    def test_perf_iris_target_num_1(self):
        test_params = [
            ("sepal_width", 0.3455068740271313),
            ("sepal_length", 0.7022285952594068),
            ("petal_width", 0.27838821814150105),
            ("petal_length", 0.5623610939600994)
        ]
        for target, ulimit in test_params:
            with self.subTest(f"target:iris({target})"):
                df = self._inject_null("iris", target)
                repaired_df = self._build_model(df).run()
                rmse = self._compute_rmse(repaired_df, "iris_clean")
                logging.info(f"target:iris({target}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.01)

    @unittest.skip(reason="")
    def test_perf_iris_target_num_2(self):
        test_params = [
            ("sepal_width", "sepal_length", 0.6595452979136459),
            ("sepal_length", "petal_width", 0.43981530214397946),
            ("petal_width", "petal_length", 0.771159840759359),
            ("petal_length", "sepal_width", 0.7145802963978227)
        ]
        for target1, target2, ulimit in test_params:
            with self.subTest(f"target:iris({target1},{target2})"):
                df = self._inject_null("iris", f"{target1},{target2}")
                repaired_df = self._build_model(df).run()
                rmse = self._compute_rmse(repaired_df, "iris_clean")
                logging.info(f"target:iris({target1},{target2}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.01)

    @unittest.skip(reason="")
    def test_perf_boston_target_num_1(self):
        test_params = [
            ("NOX", 0.03053089633885037),
            ("PTRATIO", 0.601539549909125),
            ("TAX", 26.792889334701275),
            ("INDUS", 1.4018760668798051)
        ]
        for target, ulimit in test_params:
            with self.subTest(f"target:boston({target})"):
                df = self._inject_null("boston", target)
                repaired_df = self._build_model(df).run()
                rmse = self._compute_rmse(repaired_df, "boston_clean")
                logging.info(f"target:boston({target}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.01)

    @unittest.skip(reason="")
    def test_perf_boston_target_num_2(self):
        test_params = [
            ("NOX", "PTRATIO", 0.4691041696958255),
            ("PTRATIO", "TAX", 56.96715426988806),
            ("TAX", "INDUS", 21.80912628903229),
            ("INDUS", "NOX", 1.270566665510965)
        ]
        for target1, target2, ulimit in test_params:
            with self.subTest(f"target:boston({target1},{target2})"):
                df = self._inject_null("boston", f"{target1},{target2}")
                repaired_df = self._build_model(df).run()
                rmse = self._compute_rmse(repaired_df, "boston_clean")
                logging.info(f"target:boston({target1},{target2}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.01)

    def test_perf_hospital(self):
        repair_targets = [
            "City",
            "HospitalName",
            "ZipCode",
            # All the values of "Score" column is NULL, so ignores it
            # "Score",
            "ProviderNumber",
            "Sample",
            "Address1",
            "HospitalType",
            "HospitalOwner",
            "PhoneNumber",
            "EmergencyService",
            "State",
            "Stateavg",
            "CountyName",
            "MeasureCode",
            "MeasureName",
            "Condition"
        ]
        repaired_df = self._build_model("hospital") \
            .setErrorCells("hospital_error_cells") \
            .setDiscreteThreshold(100) \
            .setTargets(repair_targets) \
            .run()

        repair_targets_set = ",".join(map(lambda x: f"'{x}'", repair_targets))
        pdf = repaired_df.join(
            self.spark.table("hospital_clean").where(f"attribute IN ({repair_targets_set})"),
            ["tid", "attribute"], "inner")
        rdf = repaired_df.join(
            self.spark.table("hospital_error_cells").where(f"attribute IN ({repair_targets_set})"),
            ["tid", "attribute"], "right_outer")

        # Computes performance numbers (precision & recall)
        #  - Precision: the fraction of correct repairs, i.e., repairs that match
        #    the ground truth, over the total number of repairs performed
        #  - Recall: correct repairs over the total number of errors
        precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
        recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
        f1 = (2.0 * precision * recall) / (precision + recall)

        msg = f"target:hospital precision:{precision} recall:{recall} f1:{f1}"
        logging.info(msg)
        self.assertTrue(precision > 0.70 and recall > 0.65 and f1 > 0.65, msg=msg)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
