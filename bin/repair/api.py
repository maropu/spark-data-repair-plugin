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

"""
A Scavenger API Set for Data Profiling & Cleaning
"""

from typing import Optional

from repair.base import *
from repair.constraints import *
from repair.misc import *
from repair.model import *


class Scavenger(ApiBase):

    __instance: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Scavenger":
        if cls.__instance == None:
            cls.__instance = super(Scavenger, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def getOrCreate() -> "Scavenger":
        return Scavenger()

    def constraints(self) -> ScavengerConstraints:
        return ScavengerConstraints().setDbName(self.db_name).setOutput(self.output)

    def repair(self) -> ScavengerRepairModel:
        return ScavengerRepairModel().setDbName(self.db_name)

    def misc(self) -> ScavengerRepairMisc:
        return ScavengerRepairMisc().setDbName(self.db_name)

