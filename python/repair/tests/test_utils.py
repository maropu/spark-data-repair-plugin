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

import re
import unittest
from typing import Dict, List, Union

from repair.utils import argtype_check


class BaseClass:
    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other):
        if isinstance(other, BaseClass):
            return self.n == other.n

        return False


class DerivedClass(BaseClass):

    def __init__(self, n: int) -> None:
        BaseClass.__init__(self, n)


@argtype_check
def test_int(v: int) -> int:
    return v


@argtype_check
def test_float(v: float) -> float:
    return v


@argtype_check
def test_str(v: str) -> str:
    return v


@argtype_check
def test_class(v: BaseClass) -> BaseClass:
    return v


@argtype_check
def test_union(v: Union[int, List[str], Dict[str, int]]) -> Union[int, List[str], Dict[str, int]]:
    return v


@argtype_check
def test_list(v: List[str]) -> List[str]:
    return v


@argtype_check
def test_class_list(v: List[BaseClass]) -> List[BaseClass]:
    return v


@argtype_check
def test_dict(v: Dict[str, int]) -> Dict[str, int]:
    return v


class UtilsTests(unittest.TestCase):

    def test_primitive_type_check(self):
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as int, got str"),
            lambda: test_int('a'))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as float, got int"),
            lambda: test_float(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as str, got int"),
            lambda: test_str(1))

        self.assertEqual(test_int(1), 1)
        self.assertEqual(test_float(2.0), 2.0)
        self.assertEqual(test_str('a'), 'a')

    def test_class_type_check(self):
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as BaseClass, got int"),
            lambda: test_class(1))

        self.assertEqual(test_class(BaseClass(1)), BaseClass(1))
        self.assertEqual(test_class(DerivedClass(1)), DerivedClass(1))

    def test_union_type_check(self):
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as int/list[str]/dict[str,int], got str"),
            lambda: test_union('a'))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as int/list[str]/dict[str,int], got list"),
            lambda: test_union([1, 2, 3]))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as int/list[str]/dict[str,int], got dict"),
            lambda: test_union({1: 1, 2: 2}))

        self.assertEqual(test_union(1), 1)
        self.assertEqual(test_union(['a', 'b']), ['a', 'b'])
        self.assertEqual(test_union({'a': 1, 'b': 2}), {'a': 1, 'b': 2})

    def test_list_type_check(self):
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as list[str], got int"),
            lambda: test_list(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as list[str], got int in elements"),
            lambda: test_list([1, 2, 3]))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as list[BaseClass], got int"),
            lambda: test_class_list(1))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as list[BaseClass], got int in elements"),
            lambda: test_class_list([1, 2, 3]))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as list[BaseClass], got str in elements"),
            lambda: test_class_list([BaseClass(1), 'a']))

        self.assertEqual(test_list(['a', 'b', 'c']), ['a', 'b', 'c'])
        self.assertEqual(test_class_list([BaseClass(1), BaseClass(2)]), [BaseClass(1), BaseClass(2)])
        self.assertNotEqual(test_class_list([BaseClass(1), BaseClass(3)]), [BaseClass(1), BaseClass(2)])
        self.assertEqual(test_class_list([DerivedClass(3)]), [DerivedClass(3)])
        self.assertNotEqual(test_class_list([DerivedClass(3)]), [DerivedClass(4)])
        self.assertEqual(test_class_list([BaseClass(1), DerivedClass(2)]), [BaseClass(1), DerivedClass(2)])

    def test_dict_type_check(self):
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as dict[str,int], got str"),
            lambda: test_dict('a'))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as dict[str,int], got list"),
            lambda: test_dict([1, 2, 3]))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as dict[str,int], got int in keys"),
            lambda: test_dict({1: 1, 2: 2}))
        self.assertRaisesRegexp(
            TypeError,
            re.escape("`v` should be provided as dict[str,int], got str in values"),
            lambda: test_dict({'a': '1', 'b': '2'}))

        self.assertEqual(test_dict({'a': 1, 'b': 2}), {'a': 1, 'b': 2})


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
