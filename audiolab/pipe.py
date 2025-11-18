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

from typing import Iterator, List, Optional, Tuple

import numpy as np

from audiolab.av.frame import pad
from audiolab.av.typing import AudioFormat, Dtype, Filter
from audiolab.reader import Graph, aformat


class AudioPipe:
    def __init__(
        self,
        in_rate: int,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        out_rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = 1024,
        fill_value: Optional[float] = None,
        always_2d: bool = True,
    ):
        self.in_rate = in_rate
        self.graph = None
        if not all([dtype is None, format is None, out_rate is None, to_mono is None]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, out_rate, to_mono))
        self.filters = filters
        self.frame_size = frame_size
        self.fill_value = fill_value
        self.always_2d = always_2d

    def push(self, frame: np.ndarray):
        if self.graph is None:
            self.graph = Graph(
                rate=self.in_rate,
                dtype=frame.dtype,
                channels=frame.shape[0],
                filters=self.filters,
                frame_size=self.frame_size,
                return_ndarray=True,
            )
        self.graph.push(frame)

    def pull(self, partial: bool = False) -> Iterator[Tuple[np.ndarray, int]]:
        for frame, rate in self.graph.pull(partial=partial):
            if self.fill_value is not None:
                frame = pad(frame, self.frame_size, self.fill_value)
            yield frame if self.always_2d else frame.squeeze(), rate
