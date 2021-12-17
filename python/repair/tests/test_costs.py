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

import unittest

from repair.costs import Levenshtein, UserDefinedUpdateCostFunction
from repair.tests.testutils import ReusedSQLTestCase


class CostsTests(ReusedSQLTestCase):

    def test_Levenshtein(self):
        f = Levenshtein()
        self.assertAlmostEqual(f.compute('111', '123'), 2.0)
        self.assertAlmostEqual(f.compute(111, 123), 2.0)
        self.assertAlmostEqual(f.compute('111', 123), 2.0)
        self.assertAlmostEqual(f.compute(111, '123'), 2.0)
        self.assertAlmostEqual(f.compute(1.11, 1.23), 2.0)
        self.assertAlmostEqual(f.compute('1.11', 1.23), 2.0)
        self.assertAlmostEqual(f.compute(1.11, '1.23'), 2.0)
        self.assertLess(f.compute('1xx%', '100%'), f.compute('1xx%', 'abcdefg'))
        # TODO: It looks like '1xx%' should be closer to '100%' than '12%' in this case
        self.assertAlmostEqual(f.compute('1xx%', '100%'), f.compute('1xx%', '12%'))
        self.assertAlmostEqual(f.compute('1xx%', '100%'), f.compute('1xx%', '1%'))
        self.assertLess(f.compute('1xx%', '100%'), f.compute('1xx%', '2%'))

    def test_UserDefinedUpdateCostFunction(self):
        import Levenshtein  # type: ignore[import]
        f = UserDefinedUpdateCostFunction(f=lambda x, y: float(1.0 / (Levenshtein.jaro(str(x), str(y)) + 0.0000001)))
        self.assertAlmostEqual(f.compute('111', '123'), 1.7999996760000587)
        self.assertAlmostEqual(f.compute(111, 123), 1.7999996760000587)
        self.assertAlmostEqual(f.compute('111', 123), 1.7999996760000587)
        self.assertAlmostEqual(f.compute(111, '123'), 1.7999996760000587)
        self.assertAlmostEqual(f.compute(1.11, 1.23), 1.499999775000034)
        self.assertAlmostEqual(f.compute('1.11', 1.23), 1.499999775000034)
        self.assertAlmostEqual(f.compute(1.11, '1.23'), 1.499999775000034)
        self.assertLess(f.compute('1xx%', '100%'), f.compute('1xx%', 'abcdefg'))
        self.assertLess(f.compute('1xx%', '100%'), f.compute('1xx%', '2%'))

        self.assertRaisesRegexp(
            ValueError,
            "`f` should take two values and return a float cost value",
            lambda: UserDefinedUpdateCostFunction(f=lambda x, y: Levenshtein.distance(str(x), str(y))))
        self.assertRaisesRegexp(
            ValueError,
            "`f` should take two values and return a float cost value",
            lambda: UserDefinedUpdateCostFunction(f=lambda x: x))


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
