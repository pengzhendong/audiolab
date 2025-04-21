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

from bv import filter
from bv.descriptor import OptionType

from audiolab.av.utils import get_template

filters = []
for name in filter.filters_available:

    globals()[name] = (
        lambda name: lambda args=None, **kwargs: (
            name,
            str(args) if args is not None else None,
            {k: str(v) for k, v in kwargs.items()},
        )
    )(name)
    globals()[name].__name__ = name

    filters.append(name)
    options = []
    _filter = filter.Filter(name)
    if _filter.options is not None:
        for opt in _filter.options:
            try:
                opt_type = opt.type
            except ValueError:
                opt_type = OptionType.STRING
            options.append(
                {
                    "name": opt.name,
                    "type": opt_type,
                    "default": opt.default,
                    "help": opt.help if opt.name != "temp" else "set temperature °C",
                }
            )
    globals()[name].__doc__ = get_template("filter").render(
        name=_filter.name, description=_filter.description, options=options
    )
