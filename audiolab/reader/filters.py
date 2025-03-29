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

from typing import Dict, Tuple, TypeAlias, Union

from av import filter
from av.option import OptionType
from jinja2 import Template

Filter: TypeAlias = Union[str, Tuple[str, str], Tuple[str, Dict[str, str]], Tuple[str, str, Dict[str, str]]]

template = """{{ description }}

Args:
{%- for option in options %}
    {{ option.name }} ({{ option.type }})
    {%- if option.help %}{{ ": " + option.help }}{% endif %}
    {%- if option.default != "" %} (default {{ option.default }}){% endif %}
{%- endfor %}

See Also:
    - Run `ffmpeg -h filter={{ name }}` for all CLI options
    - [{{ name }}](https://ffmpeg.org/ffmpeg-filters.html#{{ name }})
"""

for name in filter.filters_available:

    globals()[name] = lambda name=name, args=None, **kwargs: (
        name,
        str(args) if args is not None else None,
        {k: str(v) for k, v in kwargs.items()},
    )
    globals()[name].__name__ = name

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
    globals()[name].__doc__ = Template(template).render(
        name=_filter.name, description=_filter.description, options=options
    )
