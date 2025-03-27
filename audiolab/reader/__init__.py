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

import sys
from typing import Any, List, Optional, Tuple

from . import filters
from .reader import Reader


def load_audio(
    file: Any,
    stream_id: int = 0,
    block_size: int = sys.maxsize,
    offset: float = 0.0,
    duration: float = None,
    filters: Optional[List[Tuple[str, str]]] = None,
) -> Reader:
    reader = Reader(file, stream_id, block_size, offset, duration, filters)
    generator = reader.__iter__()
    if block_size < sys.maxsize:
        return generator
    return next(generator)


__all__ = ["Reader", "load_audio", "filters"]
