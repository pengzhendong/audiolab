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

from functools import cached_property, partial
from io import BytesIO
from typing import Any, Iterator, List, Optional

from audiolab.av import aformat, load_url
from audiolab.av.frame import pad
from audiolab.av.graph import Graph
from audiolab.av.typing import UINT32_MAX, AudioFrame, Dtype, Filter, Seconds
from audiolab.reader.backend import Backend, pyav, soundfile
from audiolab.reader.info import Info


class Reader(Info):
    def __init__(
        self,
        file: Any,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[Dtype] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = None,
        cache_url: bool = False,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
        backends: Optional[List[Backend]] = None,
    ):
        """
        Create a Reader object.

        Args:
            file: The audio file, audio url, path to audio file, bytes of audio data, etc.
            offset: The offset of the audio to load.
            duration: The duration of the audio to load.
            filters: The filters to apply to the audio.
            dtype: The data type of the audio frames.
            rate: The sample rate of the audio frames.
            to_mono: Whether to convert the audio frames to mono.
            frame_size: The frame size of the audio frames.
            cache_url: Whether to cache the audio file.
            always_2d: Whether to return 2d ndarrays even if the audio frame is mono.
            fill_value: The fill value to pad the audio to the frame size.
            backends: The backends to use.
        """
        if isinstance(file, bytes):
            file = BytesIO(file)
        elif isinstance(file, str) and "://" in file:
            if cache_url:
                file = load_url(file, cache=True)
            elif offset == 0 and duration is None:
                file = load_url(file, cache=False)

        super().__init__(file, frame_size, backends=backends)
        if isinstance(self.backend, soundfile):
            self.backend.read = partial(self.backend.read, dtype=dtype)
        self.filters = [] if filters is None else filters
        if not self.is_passthrough(dtype, rate, to_mono):
            self.filters.append(aformat(dtype, rate=rate, to_mono=to_mono))

        self.graph = None
        if len(self.filters) > 0:
            if isinstance(self.backend, pyav):
                self.backend.build_graph = partial(self.backend.build_graph, filters=self.filters)
            else:
                self.graph = Graph(
                    rate=self.rate,
                    dtype=self.dtype,
                    is_planar=self.backend.is_planar,
                    channels=self.num_channels,
                    filters=self.filters,
                    frame_size=self.frame_size,
                )
        self.offset = offset
        self._duration = duration
        self.always_2d = always_2d
        self.fill_value = fill_value

    @cached_property
    def frame_size(self) -> int:
        return self.backend.frame_size

    def __iter__(self) -> Iterator[AudioFrame]:
        for frame in self.backend.load_audio(self.offset, self._duration):
            if self.graph is None:
                rate = self.rate
                if isinstance(self.backend, pyav):
                    frame, rate = frame
                if self.fill_value is not None:
                    frame = pad(frame, self.frame_size, self.fill_value)
                yield frame if self.always_2d else frame.squeeze(), rate
            else:
                self.graph.push(frame)
                yield from self.pull()
        if self.graph is not None:
            yield from self.pull(partial=True)

    def is_passthrough(self, dtype: Optional[Dtype] = None, rate: Optional[int] = None, to_mono: bool = False) -> bool:
        passthrough = dtype is None or dtype == self.dtype
        passthrough = passthrough and (rate is None or self.rate == rate)
        passthrough = passthrough and not (to_mono and self.num_channels > 1)
        passthrough = passthrough and self.frame_size >= UINT32_MAX
        passthrough = passthrough and len(self.filters) == 0
        return passthrough

    def pull(self, partial: bool = False) -> AudioFrame:
        for frame in self.graph.pull(partial=partial):
            frame, rate = frame
            if self.fill_value is not None:
                frame = pad(frame, self.frame_size, self.fill_value)
            yield frame if self.always_2d else frame.squeeze(), rate
