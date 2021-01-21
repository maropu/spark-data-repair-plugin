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
from pyspark.sql import Row

from repair.misc import ScavengerRepairMisc


def load_testdata(spark, filename):
    fmt = os.path.splitext(filename)[1][1:]
    return spark.read.format(fmt) \
        .option("header", True) \
        .load("{}/{}".format(os.getenv("SCAVENGER_TESTDATA"), filename))


@unittest.skipIf(
    not have_pandas or not have_pyarrow,
    pandas_requirement_message or pyarrow_requirement_message)
class ScavengerRepairModelTests(ReusedSQLTestCase):

    @classmethod
    def conf(cls):
        return SparkConf() \
            .set("spark.jars", os.getenv("SCAVENGER_REPAIR_API_LIB")) \
            .set("spark.sql.crossJoin.enabled", "true") \
            .set("spark.sql.cbo.enabled", "true") \
            .set("spark.sql.statistics.histogram.enabled", "true") \
            .set("spark.sql.statistics.histogram.numBins", "254")

    @classmethod
    def setUpClass(cls):
        super(ScavengerRepairModelTests, cls).setUpClass()

        # Tunes # shuffle partitions
        num_parallelism = cls.spark.sparkContext.defaultParallelism
        cls.spark.sql(f"SET spark.sql.shuffle.partitions={num_parallelism}")

        # Loads test data
        load_testdata(cls.spark, "adult.csv").createOrReplaceTempView("adult")

    def test_splitInputTableInto(self):
        misc = ScavengerRepairMisc().setDbName("")
        df = misc.setTableName("adult").setRowId("tid").setK(3).splitInputTableInto()
        self.assertEqual(
            df.selectExpr("k").distinct().orderBy("k").collect(),
            [Row(k=0), Row(k=1), Row(k=2)])


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
