..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.


==============
repair package
==============

Repair Model APIs
-----------------

.. currentmodule:: repair.model

Interface to detect error cells in given input data and build a statistical
model to repair them.

.. autosummary::
    :toctree: apis

    RepairModel.option
    RepairModel.run
    RepairModel.setAttrMaxNumToComputeDomains
    RepairModel.setAttrStatSampleRatio
    RepairModel.setAttrStatThreshold
    RepairModel.setDbName
    RepairModel.setDiscreteThreshold
    RepairModel.setDomainThresholds
    RepairModel.setErrorCells
    RepairModel.setErrorDetectors
    RepairModel.setInput
    RepairModel.setMaximalLikelihoodRepairEnabled
    RepairModel.setModelLoggingEnabled
    RepairModel.setMaxTrainingRowNum
    RepairModel.setMaxTrainingColumnNum
    RepairModel.setTrainingDataRebalancingEnabled
    RepairModel.setMinCorrThreshold
    RepairModel.setRepairDelta
    RepairModel.setRowId
    RepairModel.setRuleBasedModelEnabled
    RepairModel.setParallelStatTrainingEnabled
    RepairModel.setSmallDomainThreshold
    RepairModel.setTableName
    RepairModel.setTargets
    RepairModel.setUpdateCostFunction

Repair Misc APIs
-----------------

.. currentmodule:: repair.misc

Interface to provide helper functionalities.

.. autosummary::
    :toctree: apis

    RepairMisc.describe
    RepairMisc.flatten
    RepairMisc.injectNull
    RepairMisc.option
    RepairMisc.options
    RepairMisc.splitInputTable
    RepairMisc.toErrorMap

