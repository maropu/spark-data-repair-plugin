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

from repair.misc import RepairMisc
from repair.tests.testutils import ReusedSQLTestCase, load_testdata


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

    def test_argtype_check(self):
        self.assertRaises(
            TypeError,
            "`key` should be provided as str, got int",
            lambda: RepairMisc().option(1, "value"))
        self.assertRaises(
            TypeError,
            "`value` should be provided as str, got int",
            lambda: RepairMisc().option("key", 1))
        self.assertRaises(
            TypeError,
            "`options` should be provided as dict[str,str], got int",
            lambda: RepairMisc().options(1))
        self.assertRaises(
            TypeError,
            "`options` should be provided as dict[str,str], got int in keys",
            lambda: RepairMisc().options({"1": "v1", 2: "v2"}))
        self.assertRaises(
            TypeError,
            "`options` should be provided as dict[str,str], got float in values",
            lambda: RepairMisc().options({"1": "v1", "2": 1.1}))

    def test_flatten(self):
        with self.tempView("tempView"):
            self.spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["tid", "v"]) \
                .createOrReplaceTempView("tempView")
            misc = RepairMisc().options({"table_name": "tempView", "row_id": "tid"})
            self.assertEqual(
                misc.flatten().orderBy("tid").collect(), [
                    Row(tid=1, attirubte="v", value="a"),
                    Row(tid=2, attirubte="v", value="b"),
                    Row(tid=3, attirubte="v", value="c")])

    def test_splitInputTable(self):
        misc = RepairMisc().options({"table_name": "adult", "row_id": "tid", "k": "3"})
        self.assertEqual(
            misc.splitInputTable().selectExpr("k").distinct().orderBy("k").collect(),
            [Row(k=0), Row(k=1), Row(k=2)])

    def test_splitInputTable_invalid_params(self):
        self.assertRaisesRegexp(
            ValueError,
            "Required options not found: table_name, row_id, k",
            lambda: RepairMisc().splitInputTable())
        self.assertRaisesRegexp(
            ValueError,
            "Option 'k' must be an integer, but 'x' found",
            lambda: RepairMisc().options({"table_name": "adult", "row_id": "tid", "k": "x"})
                                .splitInputTable())

    def test_injectNull(self):
        with self.tempView("tempView"):
            data = [(1, "a", 1), (2, "b", 1), (3, "c", 1), (4, "d", 2)]
            self.spark.createDataFrame(data, ["tid", "v1", "v2"]) \
                .createOrReplaceTempView("tempView")
            misc = RepairMisc().options(
                {"table_name": "tempView", "target_attr_list": "v1", "null_ratio": "1.0"})
            self.assertEqual(
                misc.injectNull().orderBy("tid").collect(), [
                    Row(tid=1, v1=None, v2=1),
                    Row(tid=2, v1=None, v2=1),
                    Row(tid=3, v1=None, v2=1),
                    Row(tid=4, v1=None, v2=2)])

    def test_describe(self):
        misc = RepairMisc().options({"table_name": "adult"})
        self.assertEqual(
            misc.describe().where("attrName != 'tid'").orderBy("attrName").collect(), [
                Row(attrName="Age", distinctCnt=4, min=None, max=None, nullCnt=2,
                    avgLen=5, maxLen=5, hist=None),
                Row(attrName="Country", distinctCnt=3, min=None, max=None, nullCnt=0,
                    avgLen=13, maxLen=13, hist=None),
                Row(attrName="Education", distinctCnt=7, min=None, max=None, nullCnt=0,
                    avgLen=9, maxLen=12, hist=None),
                Row(attrName="Income", distinctCnt=2, min=None, max=None, nullCnt=2,
                    avgLen=11, maxLen=11, hist=None),
                Row(attrName="Occupation", distinctCnt=7, min=None, max=None, nullCnt=0,
                    avgLen=13, maxLen=17, hist=None),
                Row(attrName="Relationship", distinctCnt=4, min=None, max=None, nullCnt=0,
                    avgLen=9, maxLen=13, hist=None),
                Row(attrName="Sex", distinctCnt=2, min=None, max=None, nullCnt=3,
                    avgLen=5, maxLen=6, hist=None)])

        with self.tempView("tempView"):
            self.spark.range(100).selectExpr("STRING(id)", "id % 9 v1", "DOUBLE(id % 17) v2") \
                .createOrReplaceTempView("tempView")
            misc = RepairMisc().option("table_name", "tempView")
            misc.describe().orderBy("attrName").show(truncate=False)
            self.assertEqual(
                misc.describe().orderBy("attrName").collect(), [
                    Row(attrName="id", distinctCnt=100, min=None, max=None, nullCnt=0,
                        avgLen=2, maxLen=2, hist=None),
                    Row(attrName="v1", distinctCnt=9, min="0", max="8", nullCnt=0, avgLen=8,
                        maxLen=8, hist=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]),
                    Row(attrName="v2", distinctCnt=17, min="0.0", max="16.0", nullCnt=0, avgLen=8,
                        maxLen=8, hist=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])])

    def test_toHistogram(self):
        with self.tempView("tempView"):
            data = [(1, "a", 1), (2, "a", 1), (3, "a", 1), (4, "a", 2)]
            self.spark.createDataFrame(data, ["tid", "v1", "v2"]) \
                .createOrReplaceTempView("tempView")

            misc = RepairMisc().options({"table_name": "tempView", "targets": "v1,v2"})
            self.assertEqual(
                misc.toHistogram().orderBy("attribute").collect(), [
                    Row(attribute="v1", histogram=[Row(value="a", cnt=4)])])

    def test_toErrormap(self):
        with self.tempView("tempView", "errorCells"):
            data = [(1, "a", 1), (2, "b", 1), (3, "c", 1), (4, "d", 2)]
            self.spark.createDataFrame(data, ["tid", "v1", "v2"]) \
                .createOrReplaceTempView("tempView")

            error_cells = [(1, "v1"), (2, "v2"), (4, "v1"), (4, "v2")]
            self.spark.createDataFrame(error_cells, ["tid", "attribute"]) \
                .createOrReplaceTempView("errorCells")

            misc = RepairMisc().options(
                {"table_name": "tempView", "row_id": "tid", "error_cells": "errorCells"})
            self.assertEqual(
                misc.toErrorMap().orderBy("tid").collect(), [
                    Row(tid=1, error_map="*-"),
                    Row(tid=2, error_map="-*"),
                    Row(tid=3, error_map="--"),
                    Row(tid=4, error_map="**")])


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
