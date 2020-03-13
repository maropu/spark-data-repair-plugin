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
from pyspark.sql import DataFrame, SparkSession

class ErrorDetector:

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.metadata = None
        self.spark = SparkSession.builder.getOrCreate()
        self.svgApi = self.spark.sparkContext._active_spark_context._jvm.ScavengerErrorDetectorApi

    def setup(self, metadata):
        self.metadata = metadata

    @abstractmethod
    def detect(self):
        raise NotImplementedError

class NullErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'NullErrorDetector')

    def detect(self):
        jdf = self.svgApi.detectNullCells('', self.metadata['discrete_attrs'], self.metadata['row_id'])
        return DataFrame(jdf, self.spark._wrapped)

class ConstraintErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'ConstraintErrorDetector')

    def detect(self):
        jdf = self.svgApi.detectErrorCellsFromConstraints(
            self.metadata['constraint_input_path'], '', self.metadata['discrete_attrs'], self.metadata['row_id'])
        return DataFrame(jdf, self.spark._wrapped)

