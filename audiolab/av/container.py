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

from collections import defaultdict
from typing import Dict, Set

import bv

from audiolab.av.typing import ContainerFormatEnum
from audiolab.av.utils import get_template

container_formats: Dict[str, bv.ContainerFormat] = {}
extension_formats: Dict[str, Set[str]] = defaultdict(set)
for name in bv.formats_available:
    container_formats[name] = bv.ContainerFormat(name)
    for extension in container_formats[name].extensions:
        extension_formats[extension].add(name)
ContainerFormat = ContainerFormatEnum("ContainerFormat", container_formats)


template = get_template("container")
for name, format in container_formats.items():
    getattr(ContainerFormat, name).__doc__ = template.render(format=format)
