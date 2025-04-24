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
from lhotse import Seconds

from audiolab.av.typing import AudioFormat, Dtype, Filter
from audiolab.reader.reader import Reader
from audiolab.reader.stream_reader import StreamReader


def load_audio(
    file: Any,
    stream_id: int = 0,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    filters: Optional[List[Filter]] = None,
    dtype: Optional[Dtype] = None,
    is_planar: bool = False,
    format: Optional[AudioFormat] = None,
    rate: Optional[int] = None,
    to_mono: bool = False,
    frame_size: Optional[int] = None,
    frame_size_ms: Optional[int] = None,
    return_ndarray: bool = True,
) -> Union[Iterator[Tuple[np.ndarray, int]], Tuple[np.ndarray, int]]:
    if frame_size_ms is None:
        frame_size = frame_size or np.iinfo(np.uint32).max
    reader = Reader(
        file,
        stream_id,
        offset,
        duration,
        filters or [],
        dtype,
        is_planar,
        format,
        rate,
        to_mono,
        frame_size,
        frame_size_ms,
        return_ndarray,
    )
    generator = reader.__iter__()
    if reader.frame_size < np.iinfo(np.uint32).max:
        return generator
    return next(generator)


__all__ = ["Reader", "StreamReader", "load_audio"]
