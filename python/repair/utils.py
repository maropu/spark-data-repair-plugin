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
import os
import time
import typing
from typing import Any


def _to_pretty_type_name(v: Any) -> str:
    if hasattr(v, "__origin__") and v.__origin__ is list:
        request_elem_type = v.__args__[0]
        return f'list[{_to_pretty_type_name(request_elem_type)}]'

    elif hasattr(v, "__origin__") and v.__origin__ is dict:
        request_key_type, request_value_type = v.__args__
        return f'dict[{_to_pretty_type_name(request_key_type)},{_to_pretty_type_name(request_value_type)}]'

    return v.__name__ if hasattr(v, "__name__") else str(v)


# TODO: Makes this method more general
def _compare_type(v: Any, annot: Any) -> bool:
    if hasattr(annot, "__origin__") and annot.__origin__ is list:
        if type(v) is not list:
            return False

        request_elem_type = annot.__args__[0]
        unmathed_elem_types = list(filter(lambda x: not _compare_type(x, request_elem_type), v))
        if len(unmathed_elem_types) > 0:
            return False

        return True

    elif hasattr(annot, "__origin__") and annot.__origin__ is dict:
        request_key_type, request_value_type = annot.__args__
        if type(v) is not dict:
            return False

        unmathed_key_types = \
            list(filter(lambda x: not _compare_type(x, request_key_type), v.keys()))
        if len(unmathed_key_types) > 0:
            return False

        unmathed_value_types = \
            list(filter(lambda x: not _compare_type(x, request_value_type), v.values()))
        if len(unmathed_value_types) > 0:
            return False

        return True

    return type(v) is annot or isinstance(v, annot)


def argtype_check(f):  # type: ignore
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):  # type: ignore
        sig = inspect.signature(f)
        for k, v in sig.bind(self, *args, **kwargs).arguments.items():
            annot = sig.parameters[k].annotation

            # Union case
            if hasattr(annot, "__origin__") and annot.__origin__ is typing.Union:
                matched_types = list(filter(lambda t: _compare_type(v, t), annot.__args__))
                if not matched_types:
                    msg = "`{}` should be provided as {}, got {}"
                    request_types = "/".join(map(lambda x: _to_pretty_type_name(x), annot.__args__))
                    raise TypeError(msg.format(k, request_types, type(v).__name__))

            # List case
            elif hasattr(annot, "__origin__") and annot.__origin__ is list:
                request_elem_type = annot.__args__[0]
                if type(v) is not list:
                    raise TypeError("`{}` should be provided as list[{}], got {}".format(
                        k, _to_pretty_type_name(request_elem_type), type(v).__name__))

                unmathed_elem_types = list(filter(lambda x: not _compare_type(x, request_elem_type), v))
                if len(unmathed_elem_types) > 0:
                    msg = "`{}` should be provided as list[{}], got {} in elements"
                    raise TypeError(msg.format(
                        k, _to_pretty_type_name(request_elem_type),
                        type(unmathed_elem_types[0]).__name__))

            # Dict case
            elif hasattr(annot, "__origin__") and annot.__origin__ is dict:
                request_key_type, request_value_type = annot.__args__
                if type(v) is not dict:
                    msg = "`{}` should be provided as dict[{},{}], got {}"
                    raise TypeError(msg.format(
                        k, _to_pretty_type_name(request_key_type),
                        _to_pretty_type_name(request_value_type),
                        type(v).__name__))

                unmathed_key_types = list(filter(lambda x: not _compare_type(x, request_key_type), v.keys()))
                if len(unmathed_key_types) > 0:
                    msg = "`{}` should be provided as dict[{},{}], got {} in keys"
                    raise TypeError(msg.format(
                        k, _to_pretty_type_name(request_key_type),
                        _to_pretty_type_name(request_value_type),
                        type(unmathed_key_types[0]).__name__))

                unmathed_value_types = list(filter(lambda x: not _compare_type(x, request_value_type), v.values()))
                if len(unmathed_value_types) > 0:
                    msg = "`{}` should be provided as dict[{},{}], got {} in values"
                    raise TypeError(msg.format(
                        k, _to_pretty_type_name(request_key_type),
                        _to_pretty_type_name(request_value_type),
                        type(unmathed_value_types[0]).__name__))

            # TODO: Supports more types, e.g., typing.Tuple

            # Other regular cases
            elif annot is not inspect._empty:
                assert not hasattr(annot, "__origin__"), \
                    "generics are not expected to reach this path"
                if not _compare_type(v, annot):
                    msg = "`{}` should be provided as {}, got {}"
                    raise TypeError(msg.format(k, _to_pretty_type_name(annot), type(v).__name__))

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


def is_testing() -> bool:
    return os.environ.get("SPARK_TESTING") is not None
