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

from audiolab._processor import AudioProcessor, build_graph_filters, validate_transforms
from audiolab.av.frame import pad, squeeze_mono
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
    """Incrementally decode an available audio source into NumPy chunks.

    ``Reader`` accepts paths, URLs, encoded bytes, and binary file-like objects.
    Use it as a context manager when decoded audio should be processed without
    loading the complete signal into memory.
    """

    def __init__(
        self,
        source: Any,
        offset: Seconds = 0.0,
        duration: Seconds | None = None,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
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
            speed: Playback speed while preserving pitch.
            pitch_shift: Pitch shift in semitones while preserving duration.
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
        validate_transforms(speed, pitch_shift)
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
        self.speed = speed
        self.pitch_shift = pitch_shift
        self.filters = []
        if filters:
            self.filters = build_graph_filters(
                filters,
                input_sample_rate=self.sample_rate,
                speed=speed,
                pitch_shift=pitch_shift,
                dtype=dtype,
                is_planar=False,
                sample_format=None,
                output_sample_rate=sample_rate,
                to_mono=to_mono,
            )

        processor_input_dtype = self.dtype
        if not self.filters and dtype is not None and isinstance(self.backend, soundfile):
            self.backend.output_dtype = dtype
            processor_input_dtype = np.dtype(dtype)

        processor_kwargs = {}
        if isinstance(self.backend, pyav):
            processor_kwargs = {
                "input_sample_format": self.backend.stream.format,
                "input_layout": self.backend.stream.layout.name,
                "input_time_base": self.backend.stream.time_base,
            }
        self._processor = AudioProcessor(
            input_sample_rate=self.sample_rate,
            input_dtype=processor_input_dtype,
            channels=self.num_channels,
            filters=self.filters,
            dtype=dtype,
            output_sample_rate=sample_rate,
            to_mono=to_mono,
            speed=speed,
            pitch_shift=pitch_shift,
            frame_size=frame_size,
            **processor_kwargs,
        )
        self.output_dtype = self._processor.output_dtype
        self.output_sample_rate = self._processor.output_sample_rate
        self.offset = offset
        self._duration = duration
        self.always_2d = always_2d
        self.fill_value = fill_value

    def close(self):
        owned_source = self._owned_source
        self._owned_source = None
        processor = getattr(self, "_processor", None)
        self._processor = None
        try:
            if processor is not None:
                processor.close()
            super().close()
        finally:
            if owned_source is not None:
                owned_source.close()

    def __iter__(self) -> Iterator[DecodedChunk]:
        for audio in self.backend.load_audio(self.offset, self._duration):
            if isinstance(audio, np.ndarray) and self.filters:
                for chunk in _iter_audio_chunks(audio):
                    self._processor.push(chunk)
                    yield from self.pull()
            else:
                self._processor.push(audio)
                yield from self.pull()
        yield from self.pull(partial=True)

    def is_passthrough(
        self, dtype: DTypeLike | None = None, sample_rate: int | None = None, to_mono: bool = False
    ) -> bool:
        return (
            len(self.filters) == 0
            and self.speed == 1
            and self.pitch_shift == 0
            and not self._needs_format_filter(dtype, sample_rate, to_mono)
        )

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
        for audio, sample_rate in self._processor.pull(partial=partial):
            if self.fill_value is not None:
                audio = pad(audio, self.frame_size, self.fill_value)
            yield audio if self.always_2d else squeeze_mono(audio), sample_rate
