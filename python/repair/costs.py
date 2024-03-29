#!/usr/bin/env python3

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

import cloudpickle
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Union


class UpdateCostFunction(metaclass=ABCMeta):

    def __init__(self, targets: List[str] = []) -> None:
        self.targets: List[str] = targets

    @abstractmethod
    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> Optional[float]:
        pass

    def compute(self, x: Optional[Union[str, int, float]], y: Optional[Union[str, int, float]]) -> Optional[float]:
        return self._compute_impl(x, y) if x and y else None


class Levenshtein(UpdateCostFunction):

    def __init__(self, targets: List[str] = []) -> None:
        UpdateCostFunction.__init__(self, targets)

    def __str__(self) -> str:
        params = f'targets={",".join(self.targets)}' if self.targets else ''
        return f'{self.__class__.__name__}({params})'

    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> Optional[float]:
        import Levenshtein
        return float(Levenshtein.distance(str(x), str(y)))


class UserDefinedUpdateCostFunction(UpdateCostFunction):

    def __init__(self, f: Callable[[str, str], float], targets: List[str] = []) -> None:
        UpdateCostFunction.__init__(self, targets)

        try:
            ret = f('x', 'y')
            if type(ret) is not float:
                raise
        except:
            raise ValueError('`f` should take two values and return a float cost value')

        # NOTE: Uses cloudpickle here because `Spark.broadcast` cannot serialize
        # lambda functions using the built-in serializer.
        self.pickled_f = cloudpickle.dumps(f)

    def __str__(self) -> str:
        params = f'targets={",".join(self.targets)}' if self.targets else ''
        return f'{self.__class__.__name__}({params})'

    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> Optional[float]:
        if not hasattr(self, "_f"):
            self._f = cloudpickle.loads(self.pickled_f)
        try:
            return float(self._f(str(x), str(y)))
        except:
            return None
