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

from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

from .filters import Filter
from .reader import Reader
from .stream_reader import StreamReader
from .utils import load_url

UINT32MAX = 2**32 - 1


def load_audio(
    file: Any,
    stream_id: int = 0,
    offset: float = 0.0,
    duration: float = None,
    filters: List[Filter] = [],
    format: Optional[str] = None,
    frame_size: Union[int, str] = UINT32MAX,
    return_ndarray: bool = True,
) -> Union[Iterator[Tuple[np.ndarray, int]], Tuple[np.ndarray, int]]:
    reader = Reader(file, stream_id, offset, duration, filters, format, frame_size, return_ndarray)
    generator = reader.__iter__()
    if frame_size < UINT32MAX:
        return generator
    return next(generator)


__all__ = ["Reader", "StreamReader", "filters", "load_audio", "load_url"]
