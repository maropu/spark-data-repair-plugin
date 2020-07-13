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

import json

from pyspark.sql import DataFrame

from repair.base import *


class ScavengerRepairMisc(ApiBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, db_name):
        super().__init__()
        self.db_name = db_name
        self.table_name = None
        self.row_id = None
        self.target_attr_list = ""
        self.null_ratio = 0.01

        # JVM interfaces for Scavenger APIs
        self.__svg_api = self.jvm.ScavengerMiscApi

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setTargetAttrList(self, target_attr_list):
        self.target_attr_list = target_attr_list
        return self

    def setNullRatio(self, null_ratio):
        self.null_ratio = null_ratio
        return self

    def injectNull(self):
        if self.table_name is None:
            raise ValueError("`setTableName` should be called before injecting NULL")

        jdf = self.__svg_api.injectNullAt(self.db_name, self.table_name, self.target_attr_list, self.null_ratio)
        return DataFrame(jdf, self.spark._wrapped)

    def toErrorMap(self, error_cells):
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before flattening")

        jdf = self.__svg_api.toErrorMap(error_cells, self.db_name, self.table_name, self.row_id)
        return DataFrame(jdf, self.spark._wrapped)

    def flatten(self):
        if self.table_name is None or self.row_id is None:
            raise ValueError("`setTableName` and `setRowId` should be called before flattening")

        jdf = self.__svg_api.flattenTable(self.db_name, self.table_name, self.row_id)
        return DataFrame(jdf, self.spark._wrapped)


class SchemaSpy(ApiBase):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(SchemaSpy, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self):
        super().__init__()
        self.driver_name = "sqlite"
        self.props = ""

        # JVM interfaces for SchemaSpy APIs
        self.spy_api = self.jvm.SchemaSpyApi

    @staticmethod
    def getOrCreate():
        return SchemaSpy()

    def setDriverName(self, driver_name):
        self.driver_name = driver_name
        return self

    def setProps(self, props):
        self.props = props
        return self

    def catalogToDataFrame(self):
        jdf = self.spy_api.catalogToDataFrame(self.db_name, self.driver_name, self.props)
        df = DataFrame(jdf, self.spark._wrapped)
        return df

    def run(self):
        result_path = self.spy_api.run(self.output, self.db_name, self.driver_name, self.props)
        return ResultBase(result_path)


# This is a method to use SchemaSpy functionality directly
def spySchema(args=""):
    sc._jvm.SchemaSpyApi.run(args)

