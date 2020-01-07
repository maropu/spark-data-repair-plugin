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
An API set for Integrity Constraint Analyzer
"""

from pyspark.sql import DataFrame, SparkSession

class SchemaSpyResult():
    """A result container class for SchemaSpy"""

    # TODO: Prohibit instantiation directly
    def __init__(self, output):
        self.output = output

    def show(self):
        import webbrowser
        webbrowser.open('file://%s/index.html' % self.output)

class SchemaSpyBase():

    def __init__(self):
        self.output = ''
        self.dbName = 'default'

    def setOutput(self, output):
        self.output = output
        return self

    def setDbName(self, dbName):
        self.dbName = dbName
        return self

class SchemaSpy(SchemaSpyBase):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(SchemaSpy, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self):
        super().__init__()
        self.driverName = 'sqlite'
        self.props = ''

    @staticmethod
    def getOrCreate():
        return SchemaSpy()

    def setDriverName(self, driverName):
        self.driverName = driverName
        return self

    def setProps(self, props):
        self.props = props
        return self

    def catalogToDataFrame(self):
        jdf = sc._jvm.ScavengerApi.catalogToDataFrame(self.dbName, self.driverName, self.props)
        spark = SparkSession.builder.getOrCreate()
        df = DataFrame(jdf, spark._wrapped)
        return df

    def run(self):
        resultPath = sc._jvm.ScavengerApi.run(self.output, self.dbName, self.driverName, self.props)
        return SchemaSpyResult(resultPath)

# Defines singleton variables for SchemaSpy
schemaspy = SchemaSpy.getOrCreate()

class ScavengerConstraints(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, dbName):
        super().__init__()
        self.output = output
        self.dbName = dbName
        self.tableName = ''

    def setTableName(self, tableName):
        self.tableName = tableName
        return self

    def infer(self):
        resultPath = sc._jvm.ScavengerApi.inferConstraints(self.output, self.dbName, self.tableName)
        return SchemaSpyResult(resultPath)

class Scavenger(SchemaSpyBase):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(Scavenger, cls).__new__(cls)
        return cls.__instance

    # TODO: Prohibit instantiation directly
    def __init__(self):
        super().__init__()
        self.inferType = 'default'

    @staticmethod
    def getOrCreate():
        return Scavenger()

    def constraints(self):
        return ScavengerConstraints(self.output, self.dbName)

    def setInferType(self, inferType):
        self.inferType = inferType
        return self

    def infer(self):
        resultPath = sc._jvm.ScavengerApi.infer(self.output, self.dbName, self.inferType)
        return SchemaSpyResult(resultPath)

# Defines singleton variables for Scavenger
scavenger = Scavenger.getOrCreate()

# This is a method to use SchemaSpy functionality directly
def spySchema(args=''):
    sc._jvm.ScavengerApi.run(args)

# TODO: Any smarter way to initialize a Spark session?
if not sc._jvm.SparkSession.getActiveSession().isDefined():
    spark.range(1)

