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

class Distance:

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.env = None

    def setup(self, env):
        self.env = env

    @abstractmethod
    def compute(self, x, y):
        raise NotImplementedError

class Levenshtein(Distance):

    def __init__(self):
        Distance.__init__(self, 'Levenshtein')

    def compute(self, x, y):
        import Levenshtein
        return float(Levenshtein.distance(x, y))

