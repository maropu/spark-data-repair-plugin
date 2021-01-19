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

from typing import Any, Optional

from pyspark.sql import SparkSession


class ApiBase():

    def __init__(self) -> None:
        self.output: Optional[str] = None
        self.db_name: str = ""

        # For Spark/JVM interactions
        self._spark = SparkSession.builder.getOrCreate()
        self._jvm = self._spark.sparkContext._active_spark_context._jvm

    def setOutput(self, output: str) -> Any:
        self.output = output
        return self

    def setDbName(self, db_name: str) -> Any:
        self.db_name = db_name
        return self

    def outputToConsole(self, msg: str) -> None:
        print(msg)


class ResultBase():

    def __init__(self, output: str) -> None:
        self.output: str = output

    def show(self) -> None:
        assert self.output is not None
        import webbrowser
        webbrowser.open(f"file://{self.output}/index.html")
