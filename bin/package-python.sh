#!/usr/bin/env bash

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

#
# Package Python codes into a zip file

set -e -o pipefail

# Sets the root directory
ROOT_DIR="$(cd "`dirname $0`"/..; pwd)"

cd ${ROOT_DIR}/python

# Remove all the __pycache__ before packaging
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf || true

# Package the Python codes
zip -r repair.zip repair  2>&1 >/dev/null && mv repair.zip lib

echo "Packaged the Python codes in ${ROOT_DIR}/python/lib/repair.zip"

