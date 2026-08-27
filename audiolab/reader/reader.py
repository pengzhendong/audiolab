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

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from audiolab.av import build_filter_chain
from audiolab.av.frame import pad
from audiolab.av.graph import Graph
from audiolab.av.typing import DecodedChunk, FilterSpec, Seconds
from audiolab.reader.backend import pyav, soundfile
from audiolab.reader.info import Info
from audiolab.reader.source import prepare_source

DEFAULT_READ_FRAMES = 65_536
MAX_FILTER_CHUNK_BYTES = 8 * 1024 * 1024


def _iter_audio_chunks(audio: np.ndarray, max_bytes: int = MAX_FILTER_CHUNK_BYTES) -> Iterator[np.ndarray]:
    bytes_per_sample = audio.shape[0] * audio.dtype.itemsize
    max_length = max(1, min(audio.shape[1], max_bytes // bytes_per_sample - 2))
    for offset in range(0, audio.shape[1], max_length):
        yield audio[:, offset : offset + max_length]


class Reader(Info):
    def __init__(
        self,
        source: Any,
        offset: Seconds = 0.0,
        duration: Seconds | None = None,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        frame_size: int | None = None,
        read_size: int = DEFAULT_READ_FRAMES,
        cache_url: bool = False,
        always_2d: bool = True,
        fill_value: float | None = None,
        backends: list[str] | None = None,
    ):
        """
        Create a Reader object.

        Args:
            source: The audio file, audio URL, path, or encoded bytes.
            offset: The offset of the audio to load.
            duration: The duration of the audio to load.
            filters: The filters to apply to the audio.
            dtype: The data type of the audio frames.
            sample_rate: The target sample rate of the decoded audio.
            to_mono: Whether to convert the audio frames to mono.
            frame_size: The frame size of the audio frames.
            read_size: Maximum number of source frames read into memory at once.
            cache_url: Whether to cache the audio file.
            always_2d: Whether to return 2d ndarrays even if the audio frame is mono.
            fill_value: The fill value to pad the audio to the frame size.
            backends: The backends to use.
        """
        if frame_size is not None and frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if read_size <= 0:
            raise ValueError("read_size must be positive")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if fill_value is not None and frame_size is None:
            raise ValueError("frame_size is required when fill_value is set")
        original_source = source
        source = prepare_source(source, offset=offset, duration=duration, cache_url=cache_url)
        self._owned_source = source if source is not original_source and hasattr(source, "close") else None
        self.frame_size = frame_size
        try:
            super().__init__(source, frame_size or read_size, backends=backends)
        except BaseException:
            if self._owned_source is not None:
                self._owned_source.close()
                self._owned_source = None
            raise
        needs_format_filter = self._needs_format_filter(
            dtype,
            sample_rate,
            to_mono,
            allow_direct_dtype=not filters,
        )
        self.filters = (
            build_filter_chain(
                filters,
                dtype=dtype,
                sample_rate=sample_rate,
                to_mono=to_mono,
                add_format=needs_format_filter,
            )
            or []
        )
        if not needs_format_filter and isinstance(self.backend, soundfile):
            self.backend.output_dtype = dtype

        self.graph = None
        if len(self.filters) > 0 and not isinstance(self.backend, pyav):
            self.graph = Graph(
                sample_rate=self.sample_rate,
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
        owned_source = self._owned_source
        self._owned_source = None
        self.graph = None
        try:
            super().close()
        finally:
            if owned_source is not None:
                owned_source.close()

    def __iter__(self) -> Iterator[DecodedChunk]:
        for audio in self.backend.load_audio(self.offset, self._duration):
            if isinstance(self.backend, pyav):
                self._ensure_graph(audio)
                self.graph.push(audio)
                yield from self.pull()
            elif self.graph is None:
                sample_rate = self.sample_rate
                if self.fill_value is not None:
                    audio = pad(audio, self.frame_size, self.fill_value)
                yield audio if self.always_2d else audio.squeeze(), sample_rate
            else:
                for chunk in _iter_audio_chunks(audio):
                    self.graph.push(chunk)
                    yield from self.pull()
        if self.graph is not None:
            yield from self.pull(partial=True)

    def _ensure_graph(self, audio_frame):
        if self.graph is not None:
            return
        self.graph = Graph(
            sample_rate=audio_frame.rate,
            sample_format=audio_frame.format,
            layout=audio_frame.layout.name,
            channels=audio_frame.layout.nb_channels,
            time_base=audio_frame.time_base,
            filters=self.filters,
            frame_size=self.frame_size,
        )

    def is_passthrough(
        self, dtype: DTypeLike | None = None, sample_rate: int | None = None, to_mono: bool = False
    ) -> bool:
        return len(self.filters) == 0 and not self._needs_format_filter(dtype, sample_rate, to_mono)

    def _needs_format_filter(
        self,
        dtype: DTypeLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        allow_direct_dtype: bool = True,
    ) -> bool:
        dtype_mismatch = dtype is not None and np.dtype(dtype) != self.dtype
        can_read_dtype_directly = allow_direct_dtype and isinstance(self.backend, soundfile)
        return (
            (dtype_mismatch and not can_read_dtype_directly)
            or (sample_rate is not None and self.sample_rate != sample_rate)
            or (to_mono and self.num_channels > 1)
        )

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        for audio, sample_rate in self.graph.pull(partial=partial):
            if self.fill_value is not None:
                audio = pad(audio, self.frame_size, self.fill_value)
            yield audio if self.always_2d else audio.squeeze(), sample_rate
