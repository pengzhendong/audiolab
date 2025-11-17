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

from functools import cached_property
from typing import Any, Iterator, List, Optional

import numpy as np

from audiolab.av import aformat
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFormat, AudioFrame, Dtype, Filter, Seconds
from audiolab.backend import pyav
from audiolab.reader.info import Info


class Reader(Info):
    def __init__(
        self,
        file: Any,
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
        cache_url: bool = False,
        return_ndarray: bool = True,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
    ):
        """
        Create a Reader object.

        Args:
            file: The audio file, audio url, path to audio file, bytes of audio data, etc.
            offset: The offset of the audio to load.
            duration: The duration of the audio to load.
            filters: The filters to apply to the audio.
            dtype: The data type of the audio frames.
            is_planar: Whether the audio frames are planar.
            format: The format of the audio frames.
            rate: The sample rate of the audio frames.
            to_mono: Whether to convert the audio frames to mono.
            frame_size: The frame size of the audio frames.
            frame_size_ms: The frame size in milliseconds of the audio frames.
            cache_url: Whether to cache the audio file.
            return_ndarray: Whether to return ndarrays.
            always_2d: Whether to return 2d ndarrays even if the audio frame is mono.
            fill_value: The fill value to pad the audio to the frame size.
        """
        super().__init__(file, frame_size, frame_size_ms, False, cache_url)

        self.offset = offset
        self._duration = duration
        if not all([dtype is None, format is None, rate is None, not to_mono]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, rate, to_mono))

        self.always_2d = always_2d
        is_planar = self.backend.is_planar if isinstance(self.backend, pyav) else False
        graph = Graph(
            rate=self.rate,
            dtype=self.dtype,
            is_planar=is_planar,
            layout=self.layout,
            filters=filters,
            frame_size=self.frame_size,
            return_ndarray=return_ndarray,
            always_2d=always_2d,
            fill_value=fill_value,
        )
        if isinstance(self.backend, pyav):
            self.backend.graph = graph
        else:
            self.graph = graph

    @cached_property
    def frame_size(self) -> int:
        return self.backend.frame_size

    def __iter__(self) -> Iterator[AudioFrame]:
        for frame in self.backend.load_audio(self.offset, self._duration):
            if isinstance(self.backend, pyav):
                yield frame
            else:
                self.graph.push(frame)
                yield from self.graph.pull()
        if isinstance(self.backend, pyav):
            yield from self.backend.graph.pull(partial=True)
        else:
            yield from self.graph.pull(partial=True)

    def load_audio(self) -> AudioFrame:
        frames, rates = zip(*self)
        assert len(set(rates)) == 1
        return np.concatenate(frames, axis=1 if self.always_2d else 0), rates[0]
