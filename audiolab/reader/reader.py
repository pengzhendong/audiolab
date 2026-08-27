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
from numpy.typing import DTypeLike

from audiolab.av import build_filter_chain
from audiolab.av.frame import pad
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFrame, Filter, Seconds
from audiolab.reader.backend import pyav, soundfile
from audiolab.reader.info import Info
from audiolab.reader.source import prepare_source

MAX_FILTER_CHUNK_BYTES = 256 * 1024 * 1024


def _iter_audio_chunks(frame, max_bytes: int = MAX_FILTER_CHUNK_BYTES):
    bytes_per_sample = frame.shape[0] * frame.dtype.itemsize
    max_length = max(1, min(frame.shape[1], max_bytes // bytes_per_sample - 2))
    for offset in range(0, frame.shape[1], max_length):
        yield frame[:, offset : offset + max_length]


class Reader(Info):
    def __init__(
        self,
        file: Any,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[DTypeLike] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = None,
        cache_url: bool = False,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
        backends: Optional[List[str]] = None,
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
        file = prepare_source(file, offset=offset, duration=duration, cache_url=cache_url)
        super().__init__(file, frame_size, backends=backends)
        needs_format_filter = self._needs_format_filter(
            dtype,
            rate,
            to_mono,
            allow_direct_dtype=not filters,
        )
        self.filters = (
            build_filter_chain(
                filters,
                dtype=dtype,
                rate=rate,
                to_mono=to_mono,
                add_format=needs_format_filter,
            )
            or []
        )
        if not needs_format_filter and isinstance(self.backend, soundfile):
            self.backend.output_dtype = dtype

        self.graph = None
        if len(self.filters) > 0:
            if not isinstance(self.backend, pyav):
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

    def close(self):
        if self.backend is None:
            return
        self.graph = None
        super().close()

    @cached_property
    def frame_size(self) -> int:
        return self.backend.frame_size

    def __iter__(self) -> Iterator[AudioFrame]:
        for frame in self.backend.load_audio(self.offset, self._duration):
            if isinstance(self.backend, pyav):
                self._ensure_graph(frame)
                self.graph.push(frame)
                yield from self.pull()
            elif self.graph is None:
                rate = self.rate
                if self.fill_value is not None:
                    frame = pad(frame, self.frame_size, self.fill_value)
                yield frame if self.always_2d else frame.squeeze(), rate
            else:
                for chunk in _iter_audio_chunks(frame):
                    self.graph.push(chunk)
                    yield from self.pull()
        if self.graph is not None:
            yield from self.pull(partial=True)

    def _ensure_graph(self, frame):
        if self.graph is not None:
            return
        self.graph = Graph(
            rate=frame.rate,
            format=frame.format,
            layout=frame.layout.name,
            channels=frame.layout.nb_channels,
            time_base=frame.time_base,
            filters=self.filters,
            frame_size=self.frame_size,
        )

    def is_passthrough(
        self, dtype: Optional[DTypeLike] = None, rate: Optional[int] = None, to_mono: bool = False
    ) -> bool:
        return len(self.filters) == 0 and not self._needs_format_filter(dtype, rate, to_mono)

    def _needs_format_filter(
        self,
        dtype: Optional[DTypeLike] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        allow_direct_dtype: bool = True,
    ) -> bool:
        dtype_mismatch = dtype is not None and np.dtype(dtype) != self.dtype
        can_read_dtype_directly = allow_direct_dtype and isinstance(self.backend, soundfile)
        return (
            (dtype_mismatch and not can_read_dtype_directly)
            or (rate is not None and self.rate != rate)
            or (to_mono and self.num_channels > 1)
        )

    def pull(self, partial: bool = False) -> AudioFrame:
        for frame in self.graph.pull(partial=partial):
            frame, rate = frame
            if self.fill_value is not None:
                frame = pad(frame, self.frame_size, self.fill_value)
            yield frame if self.always_2d else frame.squeeze(), rate
