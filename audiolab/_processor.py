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

import math
from collections import deque
from collections.abc import Iterator
from fractions import Fraction

import av
import numpy as np
import soxr
from numpy.typing import DTypeLike

from audiolab.av.filter import asetrate, atempo
from audiolab.av.frame import clip, to_ndarray
from audiolab.av.graph import Graph
from audiolab.av.processing import aformat, build_filter_chain
from audiolab.av.typing import AudioFormatLike, AudioLayoutLike, DecodedChunk, FilterSpec, GraphInput

_SOXR_DTYPES = {np.dtype(dtype) for dtype in (np.int16, np.int32, np.float32, np.float64)}
_MAX_ATEMPO_STAGES = 32


def validate_transforms(speed: float, pitch_shift: float) -> None:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be a positive finite number")
    if not math.isfinite(pitch_shift):
        raise ValueError("pitch_shift must be finite")
    pitch_ratio = _pitch_ratio(pitch_shift)
    _atempo_filters(speed / pitch_ratio)


def _pitch_ratio(pitch_shift: float) -> float:
    try:
        ratio = math.pow(2.0, pitch_shift / 12)
    except OverflowError:
        raise ValueError("pitch_shift is outside the supported range") from None
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("pitch_shift is outside the supported range")
    return ratio


def build_graph_filters(
    filters: list[FilterSpec],
    *,
    input_sample_rate: int,
    speed: float,
    pitch_shift: float,
    dtype: DTypeLike | None,
    is_planar: bool,
    sample_format: AudioFormatLike | None,
    output_sample_rate: int | None,
    to_mono: bool,
) -> list[FilterSpec]:
    """Build the compatibility path used when callers request custom FFmpeg filters."""
    chain = list(filters)
    pitch_ratio = _pitch_ratio(pitch_shift)
    if not math.isclose(pitch_ratio, 1.0):
        chain.append(asetrate(input_sample_rate * pitch_ratio))
    chain.extend(_atempo_filters(speed / pitch_ratio))
    graph_output_rate = output_sample_rate
    if graph_output_rate is None and not math.isclose(pitch_ratio, 1.0):
        graph_output_rate = input_sample_rate
    return (
        build_filter_chain(
            chain,
            dtype=dtype,
            is_planar=is_planar,
            sample_format=sample_format,
            sample_rate=graph_output_rate,
            to_mono=to_mono,
            add_format=(
                dtype is not None
                or sample_format is not None
                or output_sample_rate is not None
                or to_mono
                or not math.isclose(pitch_ratio, 1.0)
            ),
        )
        or []
    )


def _atempo_filters(factor: float) -> list[FilterSpec]:
    """Keep every stage in FFmpeg's high-quality 0.5x-2x tempo range."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("speed and pitch_shift require an unsupported tempo ratio")
    filters = []
    while factor > 2:
        if len(filters) == _MAX_ATEMPO_STAGES:
            raise ValueError("speed and pitch_shift require too many tempo stages")
        filters.append(atempo(2))
        factor /= 2
    while factor < 0.5:
        if len(filters) == _MAX_ATEMPO_STAGES:
            raise ValueError("speed and pitch_shift require too many tempo stages")
        filters.append(atempo(0.5))
        factor /= 0.5
    if not math.isclose(factor, 1.0):
        filters.append(atempo(factor))
    return filters


class _FrameBuffer:
    def __init__(self, frame_size: int | None):
        self.frame_size = frame_size
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._samples = 0
        self._retained_bytes = 0

    def push(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        if audio.shape[1] == 0:
            return
        if self.frame_size is None:
            yield audio
            return
        self._chunks.append(audio)
        self._samples += audio.shape[1]
        self._retained_bytes += audio.nbytes
        while self._samples >= self.frame_size:
            yield self._take(self.frame_size)

    def flush(self) -> Iterator[np.ndarray]:
        if self._samples:
            yield self._take(self._samples)

    def _take(self, samples: int) -> np.ndarray:
        first = self._chunks[0]
        available = first.shape[1] - self._offset
        if self._offset == 0 and available == samples:
            self._chunks.popleft()
            self._samples -= samples
            self._retained_bytes -= first.nbytes
            return first

        output = np.empty((first.shape[0], samples), dtype=first.dtype)
        written = 0
        while written < samples:
            chunk = self._chunks[0]
            count = min(samples - written, chunk.shape[1] - self._offset)
            output[:, written : written + count] = chunk[:, self._offset : self._offset + count]
            written += count
            self._offset += count
            if self._offset == chunk.shape[1]:
                self._chunks.popleft()
                self._retained_bytes -= chunk.nbytes
                self._offset = 0
        self._samples -= samples
        return output

    def clear(self) -> None:
        self._chunks.clear()
        self._offset = 0
        self._samples = 0
        self._retained_bytes = 0

    @property
    def buffered_bytes(self) -> int:
        return self._retained_bytes


class AudioProcessor:
    """Private execution plan for common transforms and custom FFmpeg filters."""

    def __init__(
        self,
        *,
        input_sample_rate: int,
        input_dtype: DTypeLike,
        channels: int,
        input_sample_format: AudioFormatLike | None = None,
        input_layout: AudioLayoutLike | None = None,
        input_time_base: Fraction | None = None,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        output_sample_rate: int | None = None,
        to_mono: bool = False,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        frame_size: int | None = None,
    ):
        validate_transforms(speed, pitch_shift)
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate or input_sample_rate
        self.input_dtype = np.dtype(input_dtype)
        self.output_dtype = np.dtype(dtype) if dtype is not None else self.input_dtype
        self.input_channels = channels
        self.output_channels = 1 if to_mono else channels
        self.to_mono = to_mono
        self.speed = speed
        self.pitch_shift = pitch_shift
        self._finalized = False
        self._frame_buffer = _FrameBuffer(frame_size)
        self._outputs: deque[np.ndarray] = deque()
        self._output_bytes = 0

        self.graph = None
        self._custom_graph = bool(filters)
        if self._custom_graph:
            self.graph = Graph(
                sample_rate=input_sample_rate,
                dtype=None if input_sample_format is not None else self.input_dtype,
                sample_format=input_sample_format,
                layout=input_layout,
                channels=channels,
                time_base=input_time_base,
                filters=filters,
                frame_size=frame_size,
            )
            self._processing_dtype = self.input_dtype
            self._resampler = None
            return

        self._processing_dtype = self.input_dtype
        if to_mono or self._processing_dtype not in _SOXR_DTYPES:
            self._processing_dtype = np.dtype(np.float32)

        pitch_ratio = _pitch_ratio(pitch_shift)
        tempo_factor = speed / pitch_ratio
        processing_filters = []
        self._direct_mono_mix = to_mono and channels <= 2
        graph_input_channels = self.output_channels
        graph_input_layout = None
        if to_mono and not self._direct_mono_mix:
            processing_filters.append(aformat(to_mono=True))
            graph_input_channels = channels
            graph_input_layout = input_layout
        processing_filters.extend(_atempo_filters(tempo_factor))
        if processing_filters:
            self.graph = Graph(
                sample_rate=input_sample_rate,
                dtype=self._processing_dtype,
                layout=graph_input_layout,
                channels=graph_input_channels,
                filters=processing_filters,
            )

        resample_rate = self.output_sample_rate / pitch_ratio
        self._resampler = None
        if not math.isclose(resample_rate, input_sample_rate):
            self._resampler = soxr.ResampleStream(
                input_sample_rate,
                resample_rate,
                self.output_channels,
                dtype=self._processing_dtype.name,
                quality="HQ",
            )

    def push(self, audio: GraphInput) -> None:
        if self._finalized:
            raise RuntimeError("Cannot push audio after processing has been finalized")
        if self._custom_graph:
            self.graph.push(audio)
            return

        if isinstance(audio, tuple):
            audio, sample_rate = audio
            if sample_rate != self.input_sample_rate:
                raise ValueError(f"Expected sample rate {self.input_sample_rate}, received {sample_rate}")
        if isinstance(audio, av.AudioFrame):
            audio = to_ndarray(audio)
        audio = np.atleast_2d(audio)
        if audio.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, received {audio.shape[0]}")
        audio = self._prepare_audio(audio)
        if self.graph is None:
            self._queue_output(audio)
            return
        self.graph.push(audio)
        for chunk, _ in self.graph.pull():
            self._queue_output(chunk)

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        if self._custom_graph:
            if self.graph is None:
                return
            completed = False
            try:
                yield from self.graph.pull(partial=partial)
                completed = True
            finally:
                if completed and partial:
                    self._finalized = True
                    self.graph = None
            return

        if partial and not self._finalized:
            self._finalized = True
            if self.graph is not None:
                for chunk, _ in self.graph.pull(partial=True):
                    self._queue_output(chunk)
                self.graph = None
            if self._resampler is not None:
                empty = np.empty((0, self.output_channels), dtype=self._processing_dtype)
                self._queue_processed(self._resampler.resample_chunk(empty, last=True).T)
                self._resampler.clear()
                self._resampler = None
            for chunk in self._frame_buffer.flush():
                self._append_output(chunk)

        while self._outputs:
            chunk = self._outputs.popleft()
            self._output_bytes -= chunk.nbytes
            yield chunk, self.output_sample_rate

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        if self._direct_mono_mix:
            audio = clip(audio, np.float32).mean(axis=0, dtype=np.float32, keepdims=True)
        elif audio.dtype != self._processing_dtype:
            audio = clip(audio, self._processing_dtype)
        return audio

    def _queue_output(self, audio: np.ndarray) -> None:
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(audio.T, last=False).T
        self._queue_processed(audio)

    def _queue_processed(self, audio: np.ndarray) -> None:
        if audio.dtype != self.output_dtype:
            audio = clip(audio, self.output_dtype)
        for chunk in self._frame_buffer.push(audio):
            self._append_output(chunk)

    def _append_output(self, audio: np.ndarray) -> None:
        self._outputs.append(audio)
        self._output_bytes += audio.nbytes

    @property
    def buffered_bytes(self) -> int:
        return self._frame_buffer.buffered_bytes + self._output_bytes

    def close(self) -> None:
        self.graph = None
        if self._resampler is not None:
            self._resampler.clear()
        self._resampler = None
        self._frame_buffer.clear()
        self._outputs.clear()
        self._output_bytes = 0
        self._finalized = True
