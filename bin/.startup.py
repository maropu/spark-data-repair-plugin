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
import math

import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import partial
from tqdm import tqdm

from pyspark.sql import DataFrame, SparkSession

# Imports for inferences
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn import Parameter, ParameterList
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Featurizer:

    __metaclass__ = ABCMeta

    def __init__(self, name, metadata, learnable=True, init_weight=1.0):
        self.name = name
        self.setup_done = False
        self.spark = SparkSession.builder.getOrCreate()
        self.metadata = metadata
        self.all_attrs = spark.table(metadata['__input_attrs']).columns
        self.total_vars = int(metadata['totalVars'])
        self.total_attrs = int(metadata['tableAttrNum'])
        self.classes = int(metadata['classes'])
        self.tensor = None
        self.learnable = learnable
        self.init_weight = init_weight

    def setup_featurizer(self, batch_size=32):
        self._batch_size = batch_size
        self.setup_done = True
        self.specific_setup()

    @abstractmethod
    def specific_setup(self):
        raise NotImplementedError

    def create_tensor(self):
        return self.tensor

    def size(self):
        return self.tensor.size()[2]

    @abstractmethod
    def feature_names(self):
        raise NotImplementedError

class InitAttrFeaturizer(Featurizer):

    def __init__(self, metadata, init_weight=1.0):
        if isinstance(init_weight, list):
            init_weight = torch.FloatTensor(init_weight)
        Featurizer.__init__(self, 'InitAttrFeaturizer', metadata, learnable=False, init_weight=init_weight)

    def specific_setup(self):
        df = self.spark.table(self.metadata['__init_attr_feature']).toPandas()
        tensors = []
        for index, row in df.iterrows():
            init_idx = int(row['init_idx'])
            attr_idx = int(row['attr_idx'])
            tensor = -1 * torch.ones(1, self.classes, self.total_attrs)
            tensor[0][init_idx][attr_idx] = 1.0
            tensors.append(tensor)

        self.tensor = torch.cat(tensors)

    def feature_names(self):
        return self.all_attrs

class FreqFeaturizer(Featurizer):

    def __init__(self, metadata):
        Featurizer.__init__(self, 'FreqFeaturizer', metadata)

    def specific_setup(self):
        df = self.spark.table(self.metadata['__freq_feature']).toPandas()
        tensors = []
        for name, group in df.groupby(['vid']):
            tensor = torch.zeros(1, self.classes, self.total_attrs)
            for index, row in group.iterrows():
                idx = int(row['idx'])
                attr_idx = int(row['attr_idx'])
                prob = float(row['prob'])
                tensor[0][idx][attr_idx] = prob
            tensors.append(tensor)

        self.tensor = torch.cat(tensors)

    def feature_names(self):
        return self.all_attrs

class OccurAttrFeaturizer(Featurizer):

    def __init__(self, metadata):
        Featurizer.__init__(self, 'OccurAttrFeaturizer', metadata)

    def specific_setup(self):
        df = self.spark.table(self.metadata['__occur_attr_feature']).toPandas()
        tensors = []
        sorted_domain = df.reset_index().sort_values(by=['vid'])[['vid', 'rv_domain_idx', 'index', 'prob']]
        for name, group in sorted_domain.groupby(['vid']):
            tensor = torch.zeros(1, self.classes, self.total_attrs * self.total_attrs)
            for index, row in group.iterrows():
                rv_domain_idx = int(row['rv_domain_idx'])
                index = int(row['index'])
                prob = float(row['prob'])
                tensor[0][rv_domain_idx][index] = prob
            tensors.append(tensor)

        self.tensor = torch.cat(tensors)

    def feature_names(self):
        return ["{} X {}".format(attr1, attr2) for attr1 in self.all_attrs for attr2 in self.all_attrs]

class ConstraintFeaturizer(Featurizer):

    def __init__(self, metadata):
        Featurizer.__init__(self, 'ConstraintFeaturizer', metadata)

    def specific_setup(self):
        df = self.spark.table(self.metadata['__constraint_feature']).toPandas()
        tensors = []
        for name, group in df.groupby(['constraintId']):
            tensor = torch.zeros(self.total_vars, self.classes, 1)
            for index, row in group.iterrows():
                vid = int(row['vid'])
                val_id = int(row['valId'])
                feat_val = float(row['violations'])
                tensor[vid][val_id][0] = feat_val
            tensors.append(tensor)

        self.tensor = F.normalize(torch.cat(tensors, 2), p=2, dim=1)

    def feature_names(self):
        return [ "fixed pred: %s, violation pred: %s" % (fixed, violation) \
            for fixed, violation in zip(self.metadata['__fixed_preds'], self.metadata['__violation_preds']) ]

class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between
    (output) classes that takes as input a tensor X with dimensions
        (batch size) X (output_dim) X (in_features)
        where:
            output_dim is the desired output dimension/# of classes
            in_features are the features with shared weights across the classes
    """

    def __init__(self, options, featurizers, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.options = options
        # Init parameters
        self.in_features = 0.0
        self.weight_list = ParameterList()
        if bias:
             self.bias_list = ParameterList()
        else:
             self.register_parameter('bias', None)
        self.output_dim = output_dim
        self.bias_flag = bias
        # Iterate over featurizer info list
        for f in featurizers:
            learnable = f.learnable
            feat_size = f.size()
            init_weight = f.init_weight
            self.in_features += feat_size
            feat_weight = Parameter(init_weight*torch.ones(1, feat_size), requires_grad=learnable)
            if learnable:
                self.reset_parameters(feat_weight)
            self.weight_list.append(feat_weight)
            if bias:
                feat_bias = Parameter(torch.zeros(1, feat_size), requires_grad=learnable)
                if learnable:
                    self.reset_parameters(feat_bias)
                self.bias_list.append(feat_bias)

    def reset_parameters(self, tensor):
        stdv = 1. / math.sqrt(tensor.size(0))
        tensor.data.uniform_(-stdv, stdv)

    def concat_weights(self):
        self.W = torch.cat([t for t in self.weight_list],-1)
        # Normalize weights.
        if self.options['weight_norm']:
            self.W = self.W.div(self.W.norm(p=2))
        # expand so we can do matrix multiplication with each cell and their max # of domain values
        self.W = self.W.expand(self.output_dim, -1)
        if self.bias_flag:
            self.B = torch.cat([t.expand(self.output_dim, -1) for t in self.bias_list],-1)

    def forward(self, X, index, mask):
        # Concatenates different featurizer weights - need to call during every pass.
        self.concat_weights()
        output = X.mul(self.W)
        if self.bias_flag:
            output += self.B
        output = output.sum(2)
        # Add our mask so that invalid domain classes for a given variable/VID
        # has a large negative value, resulting in a softmax probability
        # of de facto 0.
        output.index_add_(0, index, mask)
        return output

class RepairDataset:

    def __init__(self, metadata, tensor):
        self.spark = SparkSession.builder.getOrCreate()
        self.total_vars = int(metadata['totalVars'])
        self.classes = int(metadata['classes'])
        self.metadata = metadata
        self.tensor = tensor

    def setup(self):
        # weak labels
        df = self.spark.table(self.metadata['__weak_label']).toPandas()
        self.weak_labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        self.is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)
        for index, row in df.iterrows():
            vid = int(row['vid'])
            label = int(row['weakLabelIndex'])
            fixed = int(row['fixed'])
            clean = int(row['clean'])
            self.weak_labels[vid] = label
            self.is_clean[vid] = clean

        # variable masks
        self.var_to_domsize = {}
        df = self.spark.table(self.metadata['__var_mask']).toPandas()
        self.var_class_mask = torch.zeros(self.total_vars, self.classes)
        for index, row in df.iterrows():
            vid = int(row['vid'])
            max_class = int(row['domainSize'])
            self.var_class_mask[vid, max_class:] = -10e6
            self.var_to_domsize[vid] = max_class

    def train_dataset(self):
        train_idx = (self.weak_labels != -1).nonzero()[:,0]
        X_train = self.tensor.index_select(0, train_idx)
        Y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return (X_train, Y_train, mask_train)

    def infer_dataset(self):
        infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return (X_infer, infer_idx, mask_infer)

class RepairModel:

    def __init__(self, options, featurizers, output_dim, bias=False):
        self.options = options
        # A list of tuples (name, is_featurizer_learnable, featurizer_output_size, init_weight, feature_names (list))
        self.output_dim = output_dim
        self.model = TiedLinear(self.options, featurizers, output_dim, bias)
        self.featurizer_weights = {}

    def fit(self, X_train, Y_train, mask_train):
        n_examples, n_classes, n_features = X_train.shape

        loss = torch.nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        lr = float(self.options['learning_rate'])
        wd = float(self.options['weight_decay'])
        if self.options['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=float(self.options['momentum']), weight_decay=wd)
        else:
            optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

        lr_sched = ReduceLROnPlateau(optimizer, 'min', verbose=True, eps=1e-5, patience=5)

        batch_size = int(self.options['batch_size'])
        epochs = int(self.options['epochs'])
        for i in tqdm(range(epochs)):
            cost = 0.
            num_batches = n_examples // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.__train__(loss, optimizer, X_train[start:end], Y_train[start:end], mask_train[start:end])

            Y_pred = self.__predict__(X_train, mask_train)
            train_loss = loss.forward(Y_pred, Variable(Y_train, requires_grad=False).squeeze(1))
            logging.debug('overall training loss: %f', train_loss)
            lr_sched.step(train_loss)

            if self.options['verbose']:
                # Compute and print accuracy at the end of epoch
                grdt = Y_train.numpy().flatten()
                Y_pred = self.__predict__(X_train, mask_train)
                Y_assign = Y_pred.data.numpy().argmax(axis=1)
                logging.debug("Epoch %d, cost = %f, acc = %.2f%%",
                    i + 1, cost / num_batches, 100. * np.mean(Y_assign == grdt))

    def infer(self, X_pred, mask_pred):
        logging.info('inferring on %d examples (cells)', X_pred.shape[0])
        output = self.__predict__(X_pred, mask_pred)
        return output

    def __train__(self, loss, optimizer, X_train, Y_train, mask_train):
        X_var = Variable(X_train, requires_grad=False)
        Y_var = Variable(Y_train, requires_grad=False)
        mask_var = Variable(mask_train, requires_grad=False)

        index = torch.LongTensor(range(X_var.size()[0]))
        index_var = Variable(index, requires_grad=False)

        optimizer.zero_grad()
        # Fully-connected layer with shared parameters between output classes
        # for linear combination of input features.
        # Mask makes invalid output classes have a large negative value so
        # to zero out softmax probability.
        fx = self.model.forward(X_var, index_var, mask_var)
        # loss is CrossEntropyLoss: combines log softmax + Negative log likelihood loss.
        # Y_Var is just a single 1D tensor with value (0 - 'class' - 1) i.e.
        # index of the correct class ('class' = max domain)
        # fx is a tensor of length 'class' the linear activation going in the softmax.
        output = loss.forward(fx, Y_var.squeeze(1))
        output.backward()
        optimizer.step()
        cost = output.item()
        return cost

    def __predict__(self, X_pred, mask_pred):
        X_var = Variable(X_pred, requires_grad=False)
        index = torch.LongTensor(range(X_var.size()[0]))
        index_var = Variable(index, requires_grad=False)
        mask_var = Variable(mask_pred, requires_grad=False)
        fx = self.model.forward(X_var, index_var, mask_var)
        output = softmax(fx, 1)
        return output

    def get_featurizer_weights(self, featurizers):
        report = ""
        for i, f in enumerate(featurizers):
            this_weight = self.model.weight_list[i].data.numpy()[0]
            weight_str = "\n".join("{name} {weight}".format(name=name, weight=weight) \
                for name, weight in zip(f.feature_names(), map(str, np.around(this_weight, 3))))
            feat_name = f.name
            feat_size = f.size()
            max_w = max(this_weight)
            min_w = min(this_weight)
            mean_w = float(np.mean(this_weight))
            abs_mean_w = float(np.mean(np.absolute(this_weight)))
            # Create report
            report += "featurizer %s,size %d,max %.4f,min %.4f,avg %.4f,abs_avg %.4f,weights:\n%s\n" % (
                feat_name, feat_size, max_w, min_w, mean_w, abs_mean_w, weight_str
            )
            # Wrap in a dictionary.
            self.featurizer_weights[feat_name] = {
                'max': max_w,
                'min': min_w,
                'avg': mean_w,
                'abs_avg': abs_mean_w,
                'weights': this_weight,
                'size': feat_size
            }
        return report

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
        metadataAsJson = sc._jvm.ScavengerApi.detectErrorCells(
          self.constraintInputPath, self.dbName, self.tableName, self.rowId)
        metadata = json.loads(metadataAsJson)

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

        # Initializes all the featurizers
        for f in featurizers:
            f.setup_featurizer()

        # Prepares train & infer data sets from metadata
        tensor = F.normalize(torch.cat([ f.create_tensor() for f in featurizers ], 2), p=2, dim=1)
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
            '__input_attrs',
            '__cell_domain',
            '__init_attr_feature',
            '__freq_feature',
            '__occur_attr_feature',
            '__constraint_feature',
            '__var_mask',
            '__weak_label'
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

