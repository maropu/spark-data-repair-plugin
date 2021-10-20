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
import typing
from typing import Any


def argtype_check(f):  # type: ignore
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):  # type: ignore
        sig = inspect.signature(f)
        for k, v in sig.bind(self, *args, **kwargs).arguments.items():
            annot = sig.parameters[k].annotation

            # Union case
            if hasattr(annot, "__origin__") and annot.__origin__ is typing.Union:
                if type(v) not in annot.__args__:
                    msg = "`{}` should be provided as {}, got {}"
                    request_types = "/".join(map(lambda x: x.__name__, annot.__args__))
                    raise TypeError(msg.format(k, request_types, type(v).__name__))

            # List case
            #
            # NOTE: `typing.List[x].__origin__` is `typing.List` in Python 3.6,
            # but it is `list` in Python 3.7+
            elif hasattr(annot, "__origin__") and \
                    annot.__origin__ in (list, typing.List):
                request_elem_type = annot.__args__[0]
                if type(v) is not list:
                    msg = "`{}` should be provided as list[{}], got {}"
                    raise TypeError(msg.format(k, request_elem_type.__name__, type(v).__name__))

                if request_elem_type is not typing.Any:
                    unmathed_elem_types = \
                        list(filter(lambda x: type(x) is not request_elem_type, v))
                    if len(unmathed_elem_types) > 0:
                        msg = "`{}` should be provided as list[{}], got {} in elements"
                        raise TypeError(msg.format(
                            k, request_elem_type.__name__,
                            type(unmathed_elem_types[0]).__name__))

            # Dict case
            #
            # NOTE: `typing.Dict[k, v].__origin__` is `typing.Dict` in Python 3.6,
            # but it is `dict` in Python 3.7+
            elif hasattr(annot, "__origin__") and \
                    annot.__origin__ in (dict, typing.Dict):
                request_key_type, request_value_type = annot.__args__
                if type(v) is not dict:
                    msg = "`{}` should be provided as dict[{},{}], got {}"
                    raise TypeError(msg.format(
                        k, request_key_type.__name__,
                        request_value_type.__name__,
                        type(v).__name__))

                if request_key_type is not typing.Any:
                    unmathed_key_types = \
                        list(filter(lambda x: type(x) is not request_key_type, v.keys()))
                    if len(unmathed_key_types) > 0:
                        msg = "`{}` should be provided as dict[{},{}], got {} in keys"
                        raise TypeError(msg.format(
                            k, request_key_type.__name__,
                            request_value_type.__name__,
                            type(unmathed_key_types[0]).__name__))

                if request_key_type is not typing.Any:
                    unmathed_value_types = \
                        list(filter(lambda x: type(x) is not request_value_type, v.values()))
                    if len(unmathed_value_types) > 0:
                        msg = "`{}` should be provided as dict[{},{}], got {} in values"
                        raise TypeError(msg.format(
                            k, request_key_type.__name__,
                            request_value_type.__name__,
                            type(unmathed_value_types[0]).__name__))

            # Other regular cases
            elif annot is not inspect._empty:
                assert not hasattr(annot, "__origin__"), \
                    "generics are not expected to reach this path"
                if annot not in [type(v), typing.Any] and not isinstance(v, annot):
                    msg = "`{}` should be provided as {}, got {}"
                    raise TypeError(msg.format(k, annot.__name__, type(v).__name__))

        return f(self, *args, **kwargs)
    return wrapper


def setup_logger() -> Any:
    from logging import getLogger, NullHandler, INFO
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    logger.addHandler(NullHandler())
    return logger


def elapsed_time(f):  # type: ignore
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):  # type: ignore
        start_time = time.time()
        ret = f(self, *args, **kwargs)
        return ret, time.time() - start_time

    return wrapper
