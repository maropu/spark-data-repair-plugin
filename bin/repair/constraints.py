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

from typing import Optional

from repair.base import ApiBase, ResultBase


class ScavengerConstraints(ApiBase):

    def __init__(self) -> None:
        self.table_name: Optional[str] = None

    def setTableName(self, table_name: str) -> "ScavengerConstraints":
        self.table_name = table_name
        return self

    def infer(self) -> ResultBase:
        if self.table_name is None:
            raise ValueError("`setTableName` should be called before inferring")

        result_path = self._jvm.ScavengerApi. \
            inferConstraints(self.output, self.db_name, self.table_name)
        return ResultBase(result_path)
