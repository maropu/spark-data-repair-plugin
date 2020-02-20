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
        self.db_name = 'default'

    def setOutput(self, output):
        self.output = output
        return self

    def setDbName(self, db_name):
        self.db_name = db_name
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
        self.driver_name = 'sqlite'
        self.props = ''
        self.spy_api = sc._jvm.SchemaSpyApi

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
        spark = SparkSession.builder.getOrCreate()
        df = DataFrame(jdf, spark._wrapped)
        return df

    def run(self):
        result_path = self.spy_api.run(self.output, self.db_name, self.driver_name, self.props)
        return SchemaSpyResult(result_path)

# Defines singleton variables for SchemaSpy
schemaspy = SchemaSpy.getOrCreate()

class ScavengerConstraints(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, db_name):
        super().__init__()
        self.output = output
        self.db_name = db_name
        self.table_name = ''
        self.svg_api = sc._jvm.ScavengerApi

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def infer(self):
        result_path = self.svg_api.inferConstraints(self.output, self.db_name, self.table_name)
        return SchemaSpyResult(result_path)

class ScavengerRepairModel(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, db_name):
        self.constraint_input_path = None
        self.db_name = db_name
        self.table_name = None
        self.row_id = None
        self.discrete_thres = 80
        self.approx_cnt_Enabled = False
        self.min_corr_thres = 10.0
        self.min_attrs_to_compute_domains = 1
        self.max_attrs_to_compute_domains = 4
        self.default_max_domain_size = 4
        # self.sample_ratio = 0.80
        self.sample_ratio = 1.0
        # self.stat_thres_ratio = 0.02
        self.stat_thres_ratio = 0.0

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

        self.svg_api = sc._jvm.ScavengerRepairApi

    def setConstraints(self, constraint_input_path):
        self.constraint_input_path = constraint_input_path
        return self

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setOption(self, key, value):
        self.options[key] = value
        return self

    def infer(self):
        # Error checks first
        if (self.constraint_input_path is None or self.table_name is None or self.row_id is None):
            raise ValueError('`setConstraints`, `setTableName`, and `setRowId` should be called before doing inferences')

        # Computes various metrics for PyTorch features
        discrete_attrs = self.svg_api.filterDiscreteAttrs(self.db_name, self.table_name, self.row_id, self.discrete_thres, self.approx_cnt_Enabled)
        err_cell_view = self.svg_api.detectErrorCells(self.constraint_input_path, '', discrete_attrs, self.row_id)
        attr_stats = self.svg_api.computeAttrStats(discrete_attrs, err_cell_view, self.row_id, self.sample_ratio, self.stat_thres_ratio)
        metadata_as_json = self.svg_api.computePrerequisiteMetadata(
          discrete_attrs, attr_stats, err_cell_view, self.row_id, self.min_corr_thres,
          self.min_attrs_to_compute_domains, self.max_attrs_to_compute_domains,
          self.default_max_domain_size)

        metadata = json.loads(metadata_as_json)
        metadata['discrete_attrs'] = discrete_attrs
        metadata['err_cells'] = err_cell_view
        metadata['attr_stats'] = attr_stats

        # Gets #variables and #classes from metadata
        total_vars = int(metadata['total_vars'])
        classes = int(metadata['classes'])

        # Generates features from metadata
        featurizers = [
            InitAttrFeaturizer(metadata),
            FreqFeaturizer(metadata),
            OccurAttrFeaturizer(metadata, self.row_id),
            ConstraintFeaturizer(metadata, self.constraint_input_path, self.row_id, self.sample_ratio)
        ]

        # Prepares train & infer data sets from metadata
        tensor = create_tensor_from_featurizers(featurizers)
        dataset = RepairDataset(metadata, tensor)
        dataset.setup()

        # Builds a repair model and repairs erronous cells
        repair_model = RepairModel(self.options, featurizers, classes)
        (X_train, Y_train, mask_train) = dataset.train_dataset()
        repair_model.fit(X_train, Y_train, mask_train)
        (X_infer, infer_idx, mask_infer) = dataset.infer_dataset()
        Y_pred = repair_model.infer(X_infer, mask_infer)

        # Prints learned weights on the repair model
        report = repair_model.get_featurizer_weights(featurizers)
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
            .join(spark.table(metadata['cell_domain']), "vid") \
            .selectExpr(self.row_id, "attr_idx", "attrName", "arrays_zip(domain, dist) dist", \
              "domain[array_position(dist, array_max(dist)) - 1] inferred")

        # Releases temporary tables for metadata
        meta_tables = [
            'discrete_attrs',
            'err_cells',
            'cell_domain',
            'var_masks',
            'weak_labels',
            'attr_stats'
        ]
        for t in meta_tables:
            spark.sql("DROP VIEW IF EXISTS %s" % metadata[t])

        return repair_df

def scavenger_evaluate_repair_result(df, expected):
    evDf = df.join(expected, (df.tid == expected.tid) & (df.attrName == expected.attribute), 'inner') \
        .selectExpr('inferred = correct_val is_correct') \
        .groupBy('is_correct') \
        .count()

    evDf.join(evDf.selectExpr('sum(count) totalCount')) \
        .where("is_correct = true") \
        .selectExpr("(count / totalCount) accuracy") \
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
        return ScavengerConstraints(self.output, self.db_name)

    def repair(self):
        return ScavengerRepairModel(self.output, self.db_name)

    def setInferType(self, inferType):
        self.inferType = inferType
        return self

    def infer(self):
        result_path = sc._jvm.ScavengerApi.infer(self.output, self.db_name, self.inferType)
        return SchemaSpyResult(result_path)

# Defines singleton variables for Scavenger
scavenger = Scavenger.getOrCreate()

# This is a method to use SchemaSpy functionality directly
def spySchema(args=''):
    sc._jvm.SchemaSpyApi.run(args)

# TODO: Any smarter way to initialize a Spark session?
if not sc._jvm.SparkSession.getActiveSession().isDefined():
    spark.range(1)

# Since 3.0, `spark.sql.crossJoin.enabled` is set to true by default
spark.sql("SET spark.sql.crossJoin.enabled=true")

