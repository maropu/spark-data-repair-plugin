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

from typing import Any

from repair.misc import RepairMisc
from repair.model import RepairModel


class Delphi():
    """A Delphi API set for data repairing.

    A :class:`Delphi` has two types of API groups:

    * ``repair``: Detect errors in input data and infer correct ones from clean data.
    * ``misc``: Provide helper functionalities.

    .. versionchanged:: 0.1.0
    """

    _instance: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Delphi":
        if cls._instance is None:
            cls._instance = super(Delphi, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def getOrCreate() -> "Delphi":
        return Delphi()

    @property
    def repair(self) -> RepairModel:
        """Returns :class:`RepairModel` to repair input data.
        """
        return RepairModel()

    @property
    def misc(self) -> RepairMisc:
        """Returns :class:`RepairMisc` for misc helper functions.
        """
        return RepairMisc()

    @staticmethod
    def version() -> str:
        # TODO: Extracts a version string from the root pom.xml
        return "0.1.0-spark3.2-EXPERIMENTAL"
