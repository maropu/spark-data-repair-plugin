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

import json
import logging
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from repair.dataset import RepairDataset
from repair.featurizers import *
from repair.model import RepairModel

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

class ScavengerErrorDetector(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, dbName):
        self.constraintInputPath = ''
        self.dbName = dbName
        self.tableName = ''
        self.rowId = 'rowId'

        # Pre-defined options
        self.options = {}
        self.options['batch_size'] = '1'
        self.options['epochs'] = '10'
        self.options['optimizer'] = 'adam'
        self.options['learning_rate'] = '0.001'
        self.options['weight_decay'] = '0.01'
        self.options['momentum'] = '0.9'
        self.options['weight_norm'] = '1'
        self.options['verbose'] = '1'

    def setConstraints(self, constraintInputPath):
        self.constraintInputPath = constraintInputPath
        return self

    def setTableName(self, tableName):
        self.tableName = tableName
        return self

    def setRowId(self, rowId):
        self.rowId = rowId
        return self

    def setOption(self, key, value):
        self.options[key] = value
        return self

    def infer(self):
        svgApi = sc._jvm.ScavengerApi
        input_table_view = svgApi.prepareInputTable(self.dbName, self.tableName, self.rowId)
        err_cell_view = svgApi.detectErrorCells(self.constraintInputPath, '', input_table_view, self.rowId)
        stats_view = svgApi.computeAttrStats(input_table_view, err_cell_view, self.rowId)
        metadata_as_json = svgApi.computeMetadata(input_table_view, stats_view, err_cell_view, self.rowId)
        metadata = json.loads(metadata_as_json)
        metadata['rowId'] = self.rowId
        metadata['constraintInputPath'] = self.constraintInputPath
        metadata['__input_table_view'] = input_table_view
        metadata['__dk_cells'] = err_cell_view
        metadata['__stats_view'] = stats_view

        # Gets #variables and #classes from metadata
        total_vars = int(metadata['totalVars'])
        classes = int(metadata['classes'])

        # Generates features from metadata
        featurizers = [
            InitAttrFeaturizer(metadata),
            FreqFeaturizer(metadata),
            OccurAttrFeaturizer(metadata),
            ConstraintFeaturizer(metadata)
        ]

        # Prepares train & infer data sets from metadata
        tensor = create_tensor_from_featurizers(featurizers)
        dataset = RepairDataset(metadata, tensor)
        dataset.setup()

        # Builds a repair model and repairs erronous cells
        repairModel = RepairModel(self.options, featurizers, classes)
        (X_train, Y_train, mask_train) = dataset.train_dataset()
        repairModel.fit(X_train, Y_train, mask_train)
        (X_infer, infer_idx, mask_infer) = dataset.infer_dataset()
        Y_pred = repairModel.infer(X_infer, mask_infer)

        # Prints learned weights on the repair model
        report = repairModel.get_featurizer_weights(featurizers)
        logging.warning(report)

        # Computes a postriori distribution for each domain
        distr = []
        Y_assign = Y_pred.data.numpy().argmax(axis=1)
        domain_size = dataset.var_to_domsize
        for idx in range(Y_pred.shape[0]):
            vid = int(infer_idx[idx])
            rv_distr = list(Y_pred[idx].data.numpy())
            rv_val_idx = int(Y_assign[idx])
            rv_prob = Y_pred[idx].data.numpy().max()
            d_size = domain_size[vid]
            distr.append({'vid': vid, 'dist':[str(p) for p in rv_distr[:d_size]]})

        # Creates a dataframe for repair-inferred output
        # TODO: Use `array_sort` in v3.0
        distr_df = pd.DataFrame(data=distr)
        repair_df = spark.createDataFrame(distr_df) \
            .join(spark.table(metadata['__cell_domain']), "vid") \
            .selectExpr(self.rowId, "attr_idx", "attrName", "arrays_zip(domain, dist) dist", \
              "domain[array_position(dist, array_max(dist)) - 1] inferred")

        # Releases temporary tables for metadata
        metaTables = [
            '__input_table_view',
            '__dk_cells',
            '__cell_domain',
            '__var_mask',
            '__weak_label',
            '__stats_view'
        ]
        for t in metaTables:
            spark.sql("DROP VIEW IF EXISTS %s" % metadata[t])

        return repair_df

def scavenger_evaluate_repair_result(df, expected):
    df.join(expected, (df.tid == expected.tid) & (df.attrName == expected.attribute), "inner") \
      .selectExpr("inferred = correct_val is_correct") \
      .groupBy("is_correct").count() \
      .where("is_correct = true") \
      .selectExpr("(count / 691) accuracy") \
      .show()

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

    def repair(self):
        return ScavengerErrorDetector(self.output, self.dbName)

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

# Since 3.0, `spark.sql.crossJoin.enabled` is set to true by default
spark.sql("SET spark.sql.crossJoin.enabled=true")

