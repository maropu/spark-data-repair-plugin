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
import logging
import torch
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from pyspark.sql import DataFrame, SparkSession

class Featurizer:

    __metaclass__ = ABCMeta

    def __init__(self, name, learnable=True, init_weight=1.0, batch_size=32, bulk_collect_thres=1000):
        self.metadata = None

        self.name = name
        self.learnable = learnable
        self.init_weight = init_weight
        self.batch_size = batch_size
        self.bulk_collect_thres = bulk_collect_thres

        self.spark = SparkSession.builder.getOrCreate()
        self.svgApi = self.spark.sparkContext._active_spark_context._jvm.ScavengerRepairFeatureApi
        self.tensor = None

    def concat_tensors(self, tensors):
        return torch.cat(tensors)

    def setup(self, metadata):
        self.metadata = metadata
        self.feature_attrs = metadata['feature_attrs']
        self.total_attrs = int(len(self.feature_attrs))
        self.total_vars = int(metadata['total_vars'])
        self.classes = int(metadata['classes'])

    @abstractmethod
    def create_dataframe(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor_from_pandas(self, pdf):
        raise NotImplementedError

    @abstractmethod
    def create_tensor_from_spark(self, df):
        raise NotImplementedError

    def create_tensor(self):
        if self.tensor is None:
            df = self.create_dataframe()
            if self.bulk_collect_thres > self.total_vars:
                pdf = df.toPandas()
                self.tensor = self.concat_tensors(self.create_tensor_from_pandas(pdf))
            else:
                self.tensor = self.concat_tensors(self.create_tensor_from_spark(df))

        return self.tensor

    def size(self):
        return self.tensor.size()[2]

    @abstractmethod
    def feature_names(self):
        raise NotImplementedError

class InitAttrFeaturizer(Featurizer):

    def __init__(self, init_weight=1.0):
        if isinstance(init_weight, list):
            init_weight = torch.FloatTensor(init_weight)
        Featurizer.__init__(self, 'InitAttrFeaturizer', learnable=False, init_weight=init_weight)

    def create_dataframe(self):
        tempView = self.svgApi.createInitAttrFeatureView(self.metadata['cell_domain'])
        return self.spark.table(tempView)

    def create_tensor_from_pandas(self, pdf):
        tensors = []
        for index, row in pdf.iterrows():
            init_idx = int(row['init_idx'])
            feature_idx = int(row['feature_idx'])
            tensor = -1 * torch.ones(1, self.classes, self.total_attrs)
            tensor[0][init_idx][feature_idx] = 1.0
            tensors.append(tensor)

        return tensors

    def create_tensor_from_spark(self, df):
        tensors = []
        for row in df.toLocalIterator():
            init_idx = int(row.init_idx)
            feature_idx = int(row.feature_idx)
            tensor = -1 * torch.ones(1, self.classes, self.total_attrs)
            tensor[0][init_idx][feature_idx] = 1.0
            tensors.append(tensor)

        return tensors

    def feature_names(self):
        return self.feature_attrs

class FreqFeaturizer(Featurizer):

    def __init__(self):
        Featurizer.__init__(self, 'FreqFeaturizer')

    def create_dataframe(self):
        tempView = self.svgApi.createFreqFeatureView(
            self.metadata['discrete_attrs'],
            self.metadata['cell_domain'],
            self.metadata['err_cells']
        )
        return self.spark.table(tempView)

    def create_tensor_from_pandas(self, pdf):
        tensors = []
        for name, group in pdf.groupby(['vid']):
            tensor = torch.zeros(1, self.classes, self.total_attrs)
            for index, row in group.iterrows():
                idx = int(row['idx'])
                feature_idx = int(row['feature_idx'])
                prob = float(row['prob'])
                tensor[0][idx][feature_idx] = prob

            tensors.append(tensor)

        return tensors

    def create_tensor_from_spark(self, df):
        group_vid = None
        tensors = []
        tensor = None
        iter = df.orderBy('vid').toLocalIterator()
        while True:
            try:
                row = next(iter)
            except StopIteration:
                tensors.append(tensor)
                break

            if group_vid != row.vid:
                if tensor is not None:
                    tensors.append(tensor)
                group_vid = row.vid
                tensor = torch.zeros(1, self.classes, self.total_attrs)

            idx = int(row.idx)
            feature_idx = int(row.feature_idx)
            prob = float(row.prob)
            tensor[0][idx][feature_idx] = prob

        return tensors

    def feature_names(self):
        return self.feature_attrs

class OccurAttrFeaturizer(Featurizer):

    def __init__(self):
        Featurizer.__init__(self, 'OccurAttrFeaturizer')

    def create_dataframe(self):
        tempView = self.svgApi.createOccurAttrFeatureView(
            self.metadata['discrete_attrs'],
            ','.join(self.metadata['feature_attrs']),
            self.metadata['err_cells'],
            self.metadata['cell_domain'],
            self.metadata['attr_stats'],
            self.metadata['row_id']
        )
        return self.spark.table(tempView)

    def create_tensor_from_pandas(self, pdf):
        tensors = []
        sorted_domain = pdf.reset_index().sort_values(by=['vid'])[['vid', 'rv_domain_idx', 'idx', 'prob']]
        for name, group in sorted_domain.groupby(['vid']):
            tensor = torch.zeros(1, self.classes, self.total_attrs * self.total_attrs)
            for index, row in group.iterrows():
                rv_domain_idx = int(row['rv_domain_idx'])
                idx = int(row['idx'])
                prob = float(row['prob'])
                tensor[0][rv_domain_idx][idx] = prob

            tensors.append(tensor)

        return tensors

    def create_tensor_from_spark(self, df):
        group_vid = None
        tensors = []
        tensor = None
        iter = df.orderBy('vid').toLocalIterator()
        while True:
            try:
                row = next(iter)
            except StopIteration:
                tensors.append(tensor)
                break

            if group_vid != row.vid:
                if tensor is not None:
                    tensors.append(tensor)
                group_vid = row.vid
                tensor = torch.zeros(1, self.classes, self.total_attrs * self.total_attrs)

            rv_domain_idx = int(row.rv_domain_idx)
            idx = int(row.idx)
            prob = float(row.prob)
            tensor[0][rv_domain_idx][idx] = prob

        return tensors

    def feature_names(self):
        return ["{} X {}".format(attr1, attr2) for attr1 in self.feature_attrs for attr2 in self.feature_attrs]

class ConstraintFeaturizer(Featurizer):

    def __init__(self):
        Featurizer.__init__(self, 'ConstraintFeaturizer')
        self.constraint_feat = None

    def concat_tensors(self, tensors):
        return F.normalize(torch.cat(tensors, 2), p=2, dim=1)

    def create_dataframe(self):
        constraint_feat_as_json = self.svgApi.createConstraintFeatureView(
            self.metadata['constraint_input_path'],
            self.metadata['discrete_attrs'],
            self.metadata['err_cells'],
            self.metadata['pos_values'],
            self.metadata['row_id'],
            self.metadata['sample_ratio']
        )
        if len(constraint_feat_as_json) > 0:
            self.constraint_feat = json.loads(constraint_feat_as_json)
            return self.spark.table(self.constraint_feat['constraint_feature'])
        else:
            raise RuntimeError('valid constraint not found')

    def create_tensor_from_pandas(self, pdf):
        tensors = []
        for name, group in pdf.groupby(['constraintId']):
            tensor = torch.zeros(self.total_vars, self.classes, 1)
            for index, row in group.iterrows():
                vid = int(row['vid'])
                val_id = int(row['valId'])
                feat_val = float(row['violations'])
                tensor[vid][val_id][0] = feat_val

            tensors.append(tensor)

        return tensors

    def create_tensor_from_spark(self, df):
        group_vid = None
        tensors = []
        tensor = None
        iter = df.orderBy('constraintId').toLocalIterator()
        while True:
            try:
                row = next(iter)
            except StopIteration:
                tensors.append(tensor)
                break

            if group_vid != row.vid:
                if tensor is not None:
                    tensors.append(tensor)
                group_vid = row.vid
                tensor = torch.zeros(self.total_vars, self.classes, 1)

            vid = int(row.vid)
            val_id = int(row.valId)
            feat_val = float(row.violations)
            tensor[vid][val_id][0] = feat_val

        return tensors

    def feature_names(self):
        return [ "fixed pred: %s, violation pred: %s" % (fixed, violation) \
            for fixed, violation in zip(self.constraint_feat['fixed_preds'], self.constraint_feat['violation_preds']) ]


# A helper method to init and create a tensor from given featurizers
def create_tensor_from_featurizers(featurizers):
    _featurizers = []
    _tensors = []
    for f in featurizers:
        try:
            _tensors.append(f.create_tensor())
            _featurizers.append(f)
        except RuntimeError as e:
            logging.warning("feature '%s' removed because: %s" % (f.name, e))

    if len(_featurizers) == 0:
        raise RuntimeError('valid featurizer not found in given featurizers')

    # Drops the featurizers that couldn't work correctly
    featurizers.clear()
    featurizers.extend(_featurizers)

    return F.normalize(torch.cat(_tensors, 2), p=2, dim=1)

