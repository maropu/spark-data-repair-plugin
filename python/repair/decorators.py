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


import functools
import inspect
import time
from typing import List, Union


def argtype_check(f):  # type: ignore
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):  # type: ignore
        sig = inspect.signature(f)
        for k, v in sig.bind(self, *args, **kwargs).arguments.items():
            annot = sig.parameters[k].annotation
            if annot is not inspect._empty and type(annot) is type:
                if annot is not type(v):
                    msg = "`{}` should be provided as {}, got {}"
                    raise TypeError(msg.format(k, annot.__name__, type(v).__name__))
            elif hasattr(annot, "__origin__") and annot.__origin__ is Union:
                if type(v) not in annot.__args__:
                    msg = "`{}` should be provided as {}, got {}"
                    request_types = "/".join(map(lambda x: x.__name__, annot.__args__))
                    raise TypeError(msg.format(k, request_types, type(v).__name__))
            elif hasattr(annot, "__origin__") and annot.__origin__ is List:
                request_elem_type = annot.__args__[0]
                if type(v) is not list:
                    msg = "`{}` should be provided as list[{}], got {}"
                    raise TypeError(msg.format(k, request_elem_type.__name__, type(v).__name__))

                unmathed_elem_types = list(filter(lambda x: type(x) is not request_elem_type, v))
                if len(unmathed_elem_types) > 0:
                    msg = "`{}` should be provided as list[{}], got {} in elements"
                    raise TypeError(msg.format(
                        k, request_elem_type.__name__,
                        type(unmathed_elem_types[0]).__name__))

        return f(self, *args, **kwargs)
    return wrapper


def elapsed_time(f):  # type: ignore
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):  # type: ignore
        start_time = time.time()
        ret = f(self, *args, **kwargs)
        return ret, time.time() - start_time

    return wrapper
