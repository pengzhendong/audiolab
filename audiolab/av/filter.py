# Copyright (c) 2025 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Collection
from typing import Any

from av import filter

from audiolab.av.utils import get_template

# The undertested ``av.option`` and ``av.descriptor`` APIs, together with the
# related ``Filter`` descriptor accessor (``Filter.options``), were removed in
# PyAV 17.1.0. Fall back to an empty option list when the accessor is gone; the
# rendered docstring still points users to ``ffmpeg -h filter=<name>`` for the
# full option list, so no information is lost.
try:
    from av.option import OptionType
except ImportError:
    OptionType = None

"""
$ ffmpeg -filters
"""


class FilterManager:
    def __init__(self):
        self._filter_data: dict[str, dict[str, Any]] = {}
        self._functions: dict[str, Callable] = {}
        self._initialized: bool = False

    def _generate_filter_data(self) -> None:
        for name in filter.filters_available:
            options = []
            _filter = filter.Filter(name)
            filter_options = getattr(_filter, "options", None)
            if filter_options is not None:
                for opt in filter_options:
                    try:
                        opt_type = opt.type
                    except ValueError:
                        opt_type = OptionType.STRING if OptionType is not None else "string"
                    options.append(
                        {
                            "name": opt.name,
                            "type": opt_type,
                            "default": opt.default,
                            "help": opt.help if opt.name != "temp" else "set temperature °C",
                        }
                    )
            self._filter_data[name] = {
                "name": _filter.name,
                "description": _filter.description,
                "options": options,
            }

    def _create_filter_function(self, name: str):
        def filter_func(args=None, **kwargs):
            return (name, None if args is None else str(args), {k: str(v) for k, v in kwargs.items()})

        filter_func.__name__ = name
        return filter_func

    def _initialize_filters(self) -> None:
        if self._initialized:
            return

        self._generate_filter_data()
        for name in filter.filters_available:
            filter_func = self._create_filter_function(name)
            data = self._filter_data[name]
            filter_func.__doc__ = get_template("filter").render(
                name=data["name"], description=data["description"], options=data["options"]
            )
            self._functions[name] = filter_func

        self._initialized = True

    def __getattr__(self, name: str) -> Callable:
        self._initialize_filters()
        try:
            return self._functions[name]
        except KeyError:
            raise AttributeError(f"{name!r} is not an FFmpeg audio filter") from None

    @property
    def filters(self) -> Collection[str]:
        return filter.filters_available


_filter_manager = FilterManager()
filters = _filter_manager.filters


def __getattr__(name: str) -> Callable:
    return getattr(_filter_manager, name)
