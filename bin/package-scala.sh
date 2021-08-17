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
# Places jar files in the 'assembly' dir

set -e -o pipefail

# Sets the root directory
ROOT_DIR="$(cd "`dirname $0`"/..; pwd)"

# Loads some variables from `pom.xml`
. ${ROOT_DIR}/bin/package.sh && get_package_variables_from_pom "${ROOT_DIR}"

BUILT_PACKAGE="${ROOT_DIR}/target/${PACKAGE_JAR_NAME}"
if [ ! -e "$BUILT_PACKAGE" ]; then
  echo "${BUILT_PACKAGE} not found" 1>&2
  exit 1
fi

# Copys the found built package into the 'assembly' dir
cp ${BUILT_PACKAGE} ${ROOT_DIR}/assembly/

echo "Places the jar files in the 'assembly' dir"

