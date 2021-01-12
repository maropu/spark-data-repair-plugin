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
        self.params = None
        self.spark = SparkSession.builder.getOrCreate()

        # JVM interfaces for Scavenger APIs
        self.api = self.spark.sparkContext._active_spark_context._jvm.ScavengerErrorDetectorApi

    def setup(self, params):
        self.params = params

    def _has_param(key):
        return key in self.params and self.params[key] is not None

    @abstractmethod
    def detect(self):
        raise NotImplementedError

class NullErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'NullErrorDetector')

    def detect(self):
        jdf = self.api.detectNullCells('', self.params['input_table'], self.params['row_id'])
        return DataFrame(jdf, self.spark._wrapped)

class RegExErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'RegExErrorDetector')

    def detect(self):
        error_pattern = self.params['error_pattern'] if self._has_param('error_pattern') else ''
        jdf = self.api.detectErrorCellsFromRegEx(
            '', self.params['input_table'], self.params['row_id'], self.params['error_pattern'],
            self.params['error_cells_as_string'])
        return DataFrame(jdf, self.spark._wrapped)

class ConstraintErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'ConstraintErrorDetector')

    def detect(self):
        constraint_path = self.params['constraint_path'] if self._has_param('constraint_path') else ''
        jdf = self.api.detectErrorCellsFromConstraints(
            '', self.params['input_table'], self.params['row_id'], constraint_path)
        return DataFrame(jdf, self.spark._wrapped)

class OutlierErrorDetector(ErrorDetector):

    def __init__(self):
        ErrorDetector.__init__(self, 'OutlierErrorDetector')

    def detect(self):
        jdf = self.api.detectErrorCellsFromOutliers(
            '', self.params['input_table'], self.params['row_id'], False)
        return DataFrame(jdf, self.spark._wrapped)

