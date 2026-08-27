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

import errno
from collections.abc import Iterator
from fractions import Fraction

import av
import numpy as np
from av import filter
from numpy.typing import DTypeLike

from audiolab.av.format import get_format
from audiolab.av.frame import from_ndarray, to_ndarray
from audiolab.av.layout import standard_channel_layouts
from audiolab.av.typing import (
    UINT32_MAX,
    AudioFormatLike,
    AudioLayoutLike,
    DecodedChunk,
    FilterSpec,
    GraphInput,
)


class Graph:
    def __init__(
        self,
        template: av.AudioStream | None = None,
        sample_rate: int | None = None,
        dtype: DTypeLike | None = None,
        is_planar: bool = False,
        sample_format: AudioFormatLike | None = None,
        layout: AudioLayoutLike | None = None,
        channels: int | None = None,
        time_base: Fraction | None = None,
        filters: list[FilterSpec] | None = None,
        frame_size: int | None = None,
    ):
        if template is not None:
            sample_rate = template.sample_rate if sample_rate is None else sample_rate
            sample_format = template.format if sample_format is None else sample_format
            layout = template.layout.name if layout is None else layout
            channels = template.channels if channels is None else channels
            time_base = template.time_base if time_base is None else time_base
        if sample_rate is None:
            raise ValueError("sample_rate is required")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if sample_format is None and dtype is None:
            raise ValueError("dtype or sample_format is required")
        if frame_size is not None and frame_size <= 0:
            raise ValueError("frame_size must be positive")

        # PyAV 17.1.0 made ``av.filter.graph.Graph`` a non-subclassable Cython
        # type, so we compose instead of subclassing it.
        self._graph = filter.Graph()

        if sample_format is None:
            sample_format = get_format(dtype, is_planar)
        sample_format = sample_format.name if isinstance(sample_format, av.AudioFormat) else sample_format
        time_base = Fraction(1, sample_rate) if time_base is None else time_base
        if layout is None:
            if channels is None:
                raise ValueError("layout or channels is required")
            try:
                layout = standard_channel_layouts[channels][0]
            except KeyError:
                raise ValueError(f"Unsupported channel count: {channels}") from None
        abuffer = self._graph.add_abuffer(None, sample_rate, sample_format, layout, channels, time_base=time_base)

        nodes = [abuffer]
        if filters is not None:
            for _filter in filters:
                name, args, kwargs = (
                    (_filter, None, {})
                    if isinstance(_filter, str)
                    else ((*_filter, {}) if len(_filter) == 2 else _filter)
                )
                nodes.append(self._graph.add(name, args, **kwargs))
        nodes.append(self._graph.add("abuffersink"))
        self._graph.link_nodes(*nodes)
        self._graph.configure()

        self.frame_size = None
        if frame_size is not None:
            self.frame_size = min(frame_size, UINT32_MAX)
            self._graph.set_audio_frame_size(self.frame_size)

        self.sample_rate = sample_rate
        self.sample_format = sample_format
        self.layout = layout

    def push(self, audio: GraphInput) -> None:
        if isinstance(audio, tuple):
            audio, sample_rate = audio
            if sample_rate != self.sample_rate:
                raise ValueError(f"Expected sample rate {self.sample_rate}, received {sample_rate}")
        if isinstance(audio, np.ndarray):
            audio = from_ndarray(audio, self.sample_format, self.layout, self.sample_rate)
        self._graph.push(audio)

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        if partial:
            self._graph.push(None)
        while True:
            try:
                audio_frame = self._graph.pull()
                yield to_ndarray(audio_frame), audio_frame.rate
            except av.EOFError:
                break
            except av.FFmpegError as e:
                if e.errno != errno.EAGAIN:
                    raise
                break
