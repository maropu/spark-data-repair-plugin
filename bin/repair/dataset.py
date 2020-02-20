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

import torch
from pyspark.sql import DataFrame, SparkSession

class RepairDataset:

    def __init__(self, metadata, tensor):
        self.spark = SparkSession.builder.getOrCreate()
        self.total_vars = int(metadata['total_vars'])
        self.classes = int(metadata['classes'])
        self.metadata = metadata
        self.tensor = tensor

    def setup(self):
        # weak labels
        df = self.spark.table(self.metadata['weak_labels']).toPandas()
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
        df = self.spark.table(self.metadata['var_masks']).toPandas()
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

