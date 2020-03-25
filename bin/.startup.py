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

import datetime
import functools
import json
import logging
import pandas as pd

from pyspark.sql import DataFrame, SparkSession

from repair.dataset import RepairDataset
from repair.detectors import *
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
        self.db_name = ''

        self.spark = SparkSession.builder.getOrCreate()

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
        df = DataFrame(jdf, self.spark._wrapped)
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

class ScavengerRepairMisc(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, db_name, table_name, row_id, svg_api):
        super().__init__()
        self.db_name = db_name
        self.table_name = table_name
        self.svg_api = svg_api
        self.row_id = row_id
        self.target_attr_list = ''
        self.null_ratio = 0.01

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

    def flattenAsDataFrame(self):
        if (self.table_name is None or self.row_id is None):
            raise ValueError('`setTableName` and `setRowId` should be called before doing inferences')

        jdf = self.svg_api.flattenAsDataFrame(self.db_name, self.table_name, self.row_id)
        df = DataFrame(jdf, self.spark._wrapped)
        return df

    def injectNull(self):
        if (self.table_name is None):
            raise ValueError('`setTableName` should be called before doing inferences')

        jdf = self.svg_api.injectNullAt(self.db_name, self.table_name, self.target_attr_list, self.null_ratio)
        df = DataFrame(jdf, self.spark._wrapped)
        return df

class ScavengerRepairModel(SchemaSpyBase):

    # TODO: Prohibit instantiation directly
    def __init__(self, output, db_name):
        super().__init__()
        self.constraint_input_path = None
        self.db_name = db_name
        self.table_name = None
        self.row_id = None
        self.black_attr_list = ''
        self.discrete_thres = 80
        self.approx_cnt_Enabled = False
        self.min_corr_thres = 10.0
        self.min_attrs_to_compute_domains = 1
        self.max_attrs_to_compute_domains = 2
        self.default_max_domain_size = 4
        # self.sample_ratio = 0.80
        self.sample_ratio = 1.0
        # self.stat_thres_ratio = 0.02
        self.stat_thres_ratio = 0.0

        # Defines detectors to discover error cells
        self.detectors = [
            # NullErrorDetector(),
            ConstraintErrorDetector()
        ]

        # Defines features used for PyTorch
        self.featurizers = [
            InitAttrFeaturizer(),
            FreqFeaturizer(),
            OccurAttrFeaturizer(),
            ConstraintFeaturizer()
        ]

        # Temporary tables for metadata
        self.meta_table_names = [
            'discrete_attrs',
            'err_cells',
            'cell_domain',
            'var_masks',
            'weak_labels',
            'attr_stats',
            'repair_candidates',
            'repair_dist'
        ]

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

    def misc(self):
        return ScavengerRepairMisc(self.db_name, self.table_name, self.row_id, self.svg_api)

    def setConstraints(self, constraint_input_path):
        self.constraint_input_path = constraint_input_path
        return self

    def setTableName(self, table_name):
        self.table_name = table_name
        return self

    def setRowId(self, row_id):
        self.row_id = row_id
        return self

    def setBlackAttrList(self, black_attr_list):
        self.black_attr_list = black_attr_list
        return self

    def setAttrNumToComputeDomains(self, min, max):
        self.min_attrs_to_compute_domains = min
        self.max_attrs_to_compute_domains = max
        return self

    def setOption(self, key, value):
        self.options[key] = value
        return self

    def tempName(self, prefix):
        return '%s_%s' % (prefix, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    def detectErrorCells(self, metadata):
        # Initializes defined error detectors with the given metadata
        for d in self.detectors:
            d.setup(metadata)

        error_cells_dfs = [ d.detect() for d in self.detectors ]

        err_cells = self.tempName('err_cells_view')
        err_cells_df = functools.reduce(lambda x, y: x.union(y), error_cells_dfs)
        err_cells_df.createOrReplaceTempView(err_cells)
        return err_cells

    def exportCleanView(self, err_cells):
        clean_input_view = self.tempName('clean_input_view')
        logging.warning("exports a clean input view as '%s'..." % clean_input_view)
        clean_df = DataFrame(self.svg_api.filterCleanRows(self.db_name, self.table_name, self.row_id, err_cells), self.spark._wrapped)
        clean_df.createOrReplaceTempView(clean_input_view)

    def exportRepairedDebugView(self, metadata):
        repaired_debug_view = self.tempName('repaired_debug_view')
        logging.warning("exports a repaired debug view as '%s'..." % repaired_debug_view)

        input_df = DataFrame(self.svg_api.flattenAsDataFrame(self.db_name, self.table_name, self.row_id), self.spark._wrapped)
        input_df.join(self.spark.table(metadata['repair_candidates']), [self.row_id, 'attribute'], 'left_outer') \
            .selectExpr(self.row_id, "attribute", "IF(isnotnull(inferred), inferred, val) AS val", \
                "isnotnull(inferred) AS replaced", "val AS orig_val", "dist") \
            .createOrReplaceTempView(repaired_debug_view)

    def cleanupMetadataViews(self):
        for t in self.meta_table_names:
            self.spark.sql("DROP VIEW IF EXISTS %s" % t)

    def run(self):
        if (self.constraint_input_path is None or self.table_name is None or self.row_id is None):
            raise ValueError('`setConstraints`, `setTableName`, and `setRowId` should be called before doing inferences')

        # Metadata used to repair a given table
        metadata = {}
        metadata['row_id'] = self.row_id
        metadata['constraint_input_path'] = self.constraint_input_path
        metadata['sample_ratio'] = self.sample_ratio
        metadata['discrete_attrs'] = self.svg_api.filterDiscreteAttrs(
            self.db_name, self.table_name, self.row_id, self.black_attr_list, self.discrete_thres,
            self.approx_cnt_Enabled)

        # Detects error cells
        metadata['err_cells'] = self.detectErrorCells(metadata)
        if self.spark.table(metadata['err_cells']).count() == 0:
            self.cleanupMetadataViews()

            input_name = '`%s`.`%s`' % (self.db_name, self.table_name) \
                if not self.db_name else '`%s`' % self.table_name
            return self.spark.table(input_name)

        # Exports the clean view for debugging
        self.exportCleanView(metadata['err_cells'])

        # Computes various metrics for PyTorch features
        metadata['attr_stats'] = self.svg_api.computeAttrStats(
            metadata['discrete_attrs'], metadata['err_cells'], self.row_id, self.sample_ratio, self.stat_thres_ratio)
        metadata_as_json = self.svg_api.computePrerequisiteMetadata(
            metadata['discrete_attrs'], metadata['attr_stats'], metadata['err_cells'],
            self.row_id, self.min_corr_thres, self.min_attrs_to_compute_domains,
            self.max_attrs_to_compute_domains,
            self.default_max_domain_size)

        metadata.update(json.loads(metadata_as_json))

        # Gets #variables and #classes from metadata
        total_vars = int(metadata['total_vars'])
        classes = int(metadata['classes'])

        # Initializes defined features with the given metadata
        for f in self.featurizers:
            f.setup(metadata)

        # Prepares train & infer data sets from metadata
        tensor = create_tensor_from_featurizers(self.featurizers)
        dataset = RepairDataset(metadata, tensor)
        dataset.setup()

        # Builds a repair model and repairs erronous cells
        repair_model = RepairModel(self.options, self.featurizers, classes)
        (X_train, Y_train, mask_train) = dataset.train_dataset()
        repair_model.fit(X_train, Y_train, mask_train)
        (X_infer, infer_idx, mask_infer) = dataset.infer_dataset()
        Y_pred = repair_model.infer(X_infer, mask_infer)

        # Prints learned weights on the repair model
        report = repair_model.get_featurizer_weights(self.featurizers)
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
        metadata['repair_dist'] = 'repair_dist_view_20200313'
        self.spark.createDataFrame(pd.DataFrame(data=distr)).createOrReplaceTempView(metadata['repair_dist'])
        metadata['repair_candidates'] = self.svg_api.createRepairCandidates(
            metadata['discrete_attrs'], metadata['cell_domain'], metadata['repair_dist'], self.row_id)

        # Exports the repair view for debugging
        self.exportRepairedDebugView(metadata)

        clean_df = self.svg_api.repairTableFrom(
            metadata['repair_candidates'], self.db_name, self.table_name, self.row_id)

        self.cleanupMetadataViews()

        return clean_df

def scavenger_evaluate_repair_result(df, expected, row_id):
    evDf = df.join(expected, (df[row_id] == expected[row_id]) & (df.attribute == expected.attribute), 'inner')

    totalErrCellDf = evDf.where('orig_val <=> correct_val') \
        .selectExpr('COUNT(1) totalErrCellCount') \
        .join(evDf.where('replaced AND val <=> correct_val').selectExpr('COUNT(1) count')) \
        .selectExpr("(count / totalErrCellCount) recall") \
        .show()

    precisionDf = evDf.where('replaced').selectExpr('val = correct_val is_correct') \
        .groupBy('is_correct') \
        .count()
    precisionDf.join(precisionDf.selectExpr('SUM(count) totalRepairCount')) \
        .where("is_correct = true") \
        .selectExpr("(count / totalRepairCount) precision") \
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

