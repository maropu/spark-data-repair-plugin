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
from repair.errors import ConstraintErrorDetector, NullErrorDetector, RegExErrorDetector
from repair.tests.requirements import have_pandas, have_pyarrow, \
    pandas_requirement_message, pyarrow_requirement_message
from repair.tests.testutils import ReusedSQLTestCase, load_testdata


def _setup_logger(logfile: str):
    from logging import getLogger, FileHandler, Formatter, DEBUG, INFO
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = FileHandler(logfile)
    fh.setLevel(INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# Logger to dump evaluation metrics
_logger = _setup_logger(f"{os.getenv('REPAIR_TESTDATA')}/test-model-perf.log")


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)  # type: ignore
class RepairModelPerformanceTests(ReusedSQLTestCase):

    @classmethod
    def conf(cls):
        return SparkConf() \
            .set("spark.master", "local[*]") \
            .set("spark.driver.memory", "6g") \
            .set("spark.jars", os.getenv("REPAIR_API_LIB")) \
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
        load_testdata(cls.spark, "iris.csv").createOrReplaceTempView("iris")

        boston_schema = "tid int, CRIM double, ZN int, INDUS double, CHAS string, " \
            "NOX double, RM double, AGE double, DIS double, RAD string, TAX int, " \
            "PTRATIO double, B double, LSTAT double"
        load_testdata(cls.spark, "boston_clean.csv").createOrReplaceTempView("boston_clean")
        load_testdata(cls.spark, "boston.csv", boston_schema) \
            .createOrReplaceTempView("boston")

        load_testdata(cls.spark, "hospital_clean.csv").createOrReplaceTempView("hospital_clean")
        load_testdata(cls.spark, "hospital.csv").createOrReplaceTempView("hospital")
        load_testdata(cls.spark, "hospital_error_cells.csv") \
            .createOrReplaceTempView("hospital_error_cells")

    @classmethod
    def tearDownClass(cls):
        super(ReusedSQLTestCase, cls).tearDownClass()

    # TODO: Needs to make statical model behaviour deterministic
    def _build_model(self, input):
        return RepairModel() \
            .setInput(input) \
            .setRowId("tid") \
            .setErrorDetectors([NullErrorDetector()]) \
            .option("model.hp.no_progress_loss", "150")

    def _compute_rmse(self, repaired_df, expected):
        # Compares predicted values with the correct ones
        cmp_df = repaired_df.join(self.spark.table(expected), ["tid", "attribute"], "inner")
        n = repaired_df.count()
        return cmp_df.selectExpr(f"sqrt(sum(pow(correct_val - repaired, 2.0)) / {n}) rmse") \
            .collect()[0] \
            .rmse

    def test_perf_iris_target_num_1(self):
        test_params = [
            ("sepal_width", 0.23277956498564178),
            ("sepal_length", 0.3980215999372857),
            ("petal_width", 0.43393250942914935),
            ("petal_length", 0.6786748681618405)
        ]
        for target, ulimit in test_params:
            with self.subTest(f"target:iris({target})"):
                repaired_df = self._build_model("iris").setTargets([target]).run()
                rmse = self._compute_rmse(repaired_df, "iris_clean")
                _logger.info(f"target:iris({target}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.10)

    def test_perf_iris_target_num_2(self):
        test_params = [
            ("sepal_width", "sepal_length", 0.3355876190363502),
            ("sepal_length", "petal_width", 0.38612750734279966),
            ("petal_width", "petal_length", 0.5277536933887835),
            ("petal_length", "sepal_width", 0.46662799458587995)
        ]
        for target1, target2, ulimit in test_params:
            with self.subTest(f"target:iris({target1},{target2})"):
                repaired_df = self._build_model("iris").setTargets([target1, target2]).run()
                rmse = self._compute_rmse(repaired_df, "iris_clean")
                _logger.info(f"target:iris({target1},{target2}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.10)

    def test_perf_boston_target_num_1(self):
        test_params = [
            ("CRIM", 6.134364848429722),
            ("RAD", 0.9903379376602871),
            ("TAX", 38.55947786645111),
            ("LSTAT", 3.31145213404028)
        ]
        for target, ulimit in test_params:
            with self.subTest(f"target:boston({target})"):
                repaired_df = self._build_model("boston").setTargets([target]).run()
                rmse = self._compute_rmse(repaired_df, "boston_clean")
                _logger.info(f"target:boston({target}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.10)

    def test_perf_boston_target_num_2(self):
        test_params = [
            ("CRIM", "RAD", 3.871610580555785),
            ("RAD", "TAX", 56.96715426988806),
            ("TAX", "LSTAT", 26.66078638300166),
            ("LSTAT", "CRIM", 4.649152759148939)
        ]
        for target1, target2, ulimit in test_params:
            with self.subTest(f"target:boston({target1},{target2})"):
                repaired_df = self._build_model("boston").setTargets([target1, target2]).run()
                rmse = self._compute_rmse(repaired_df, "boston_clean")
                _logger.info(f"target:boston({target1},{target2}) RMSE:{rmse}")
                self.assertLess(rmse, ulimit + 0.10)

    def test_perf_hospital(self):
        repair_targets = [
            "City",
            "HospitalName",
            "ZipCode",
            "Score",
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

        rule_based_model_targets = [
            "EmergencyService",
            "Condition",
            "City",
            "MeasureCode",
            "HospitalName",
            "ZipCode",
            "Address1",
            "HospitalOwner",
            "ProviderNumber",
            "CountyName",
            "MeasureName"
        ]

        weighted_prob_targets = [
            "Score",
            "Sample"
        ]

        # Sets params for a hospital repair model
        constraint_path = "{}/hospital_constraints.txt".format(os.getenv("REPAIR_TESTDATA"))
        error_detectors = [
            ConstraintErrorDetector(constraint_path, targets=rule_based_model_targets),
            RegExErrorDetector("Sample", "^[0-9]{1,3} patients$"),
            RegExErrorDetector("Score", "^[0-9]{1,3}%$")
        ]

        import Levenshtein
        from repair.costs import UserDefinedUpdateCostFunction

        def compute_distance(x, y):
            return float(abs(len(str(x)) - len(str(y))) + Levenshtein.distance(str(x), str(y)))

        repaired_df = self._build_model("hospital") \
            .setErrorCells("hospital_error_cells") \
            .setDiscreteThreshold(400) \
            .setTargets(repair_targets) \
            .setErrorDetectors(error_detectors) \
            .setRepairByRules(True) \
            .setUpdateCostFunction(UserDefinedUpdateCostFunction(f=compute_distance, targets=weighted_prob_targets)) \
            .option("model.rule.repair_by_regex.disabled", "") \
            .option("model.rule.merge_threshold", "2.0") \
            .option("model.max_training_column_num", "128") \
            .option("model.hp.no_progress_loss", "10") \
            .option("repair.pmf.cost_weight", "0.1") \
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
        #
        # NOTE: Since some correct values are NULL in the 'Score' column, we ignore them here
        precision = pdf.where("correct_val IS NULL OR repaired <=> correct_val").count() / pdf.count()
        recall = rdf.where("correct_val IS NULL OR repaired <=> correct_val").count() / rdf.count()
        f1 = (2.0 * precision * recall) / (precision + recall)

        def hospital_incorrect_cell_hist(rdf) -> str:
            pdf = rdf.where('NOT(repaired <=> correct_val)').groupBy('attribute').count().toPandas()
            return ', '.join(map(lambda r: f'{r.attribute}:{r.count}', pdf.itertuples()))

        msg = f"target:hospital precision:{precision} recall:{recall} f1:{f1} " \
            f"stats:{hospital_incorrect_cell_hist(rdf)}"
        _logger.info(msg)
        self.assertTrue(precision > 0.95 and recall > 0.95 and f1 > 0.95, msg=msg)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
