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

from abc import ABCMeta, abstractmethod
from typing import Union


class UpdateCostFunction(metaclass=ABCMeta):

    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> float:
        pass

    def compute(self, x: Union[str, int, float], y: Union[str, int, float]) -> float:
        cost = self._compute_impl(x, y)
        assert type(cost) is float
        return cost


class NoCost(UpdateCostFunction):

    def __init__(self) -> None:
        UpdateCostFunction.__init__(self, 'NoCost')

    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> float:
        return 0.0


class Levenshtein(UpdateCostFunction):

    def __init__(self) -> None:
        UpdateCostFunction.__init__(self, 'Levenshtein')

    def _compute_impl(self, x: Union[str, int, float], y: Union[str, int, float]) -> float:
        import Levenshtein  # type: ignore[import]
        return float(Levenshtein.distance(str(x), str(y)))
