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

import numpy as np
from numpy.typing import DTypeLike

from audiolab._processor import AudioProcessor, build_graph_filters, validate_transforms
from audiolab.av.frame import pad, squeeze_mono
from audiolab.av.typing import FilterSpec

DEFAULT_MAX_BUFFERED_BYTES = 32 * 1024 * 1024


class AudioPipe:
    """Incrementally transform channels-first NumPy PCM chunks.

    Push consecutive input chunks with :meth:`push`, drain available output
    with :meth:`pull`, and call ``pull(partial=True)`` exactly once at end of
    input. Resampling, channel mixing, dtype conversion, speed, and pitch are
    selected through high-level arguments; custom filters remain an advanced
    opt-in path.
    """

    def __init__(
        self,
        input_sample_rate: int,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        output_sample_rate: int | None = None,
        to_mono: bool = False,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        frame_size: int | None = 1024,
        fill_value: float | None = None,
        always_2d: bool = True,
        max_buffered_bytes: int | None = DEFAULT_MAX_BUFFERED_BYTES,
    ):
        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive")
        if output_sample_rate is not None and output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        if frame_size is not None and frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if fill_value is not None and frame_size is None:
            raise ValueError("frame_size is required when fill_value is set")
        if max_buffered_bytes is not None and max_buffered_bytes <= 0:
            raise ValueError("max_buffered_bytes must be positive")
        validate_transforms(speed, pitch_shift)
        self.input_sample_rate = input_sample_rate
        self.filters = None
        if filters:
            self.filters = build_graph_filters(
                filters,
                input_sample_rate=input_sample_rate,
                speed=speed,
                pitch_shift=pitch_shift,
                dtype=dtype,
                is_planar=False,
                sample_format=None,
                output_sample_rate=output_sample_rate,
                to_mono=to_mono,
            )
        self.dtype = dtype
        self.output_sample_rate = output_sample_rate
        self.to_mono = to_mono
        self.speed = speed
        self.pitch_shift = pitch_shift
        self.frame_size = frame_size
        self.fill_value = fill_value
        self.always_2d = always_2d
        self.max_buffered_bytes = max_buffered_bytes
        self._buffered_bytes = 0
        self._finalized = False
        self._closed = False
        self._processor = None

    def push(self, audio: np.ndarray) -> None:
        if self._finalized or self._closed:
            raise RuntimeError("Cannot push audio after the pipe has been finalized or closed")
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim != 2 or audio.shape[0] == 0:
            raise ValueError("audio must have shape (samples,) or (channels, samples)")
        buffered_bytes = self.buffered_bytes + audio.nbytes
        if self.max_buffered_bytes is not None and buffered_bytes > self.max_buffered_bytes:
            raise BufferError("AudioPipe buffer limit exceeded; call pull() before pushing more audio")
        if self._processor is None:
            self._processor = AudioProcessor(
                input_sample_rate=self.input_sample_rate,
                input_dtype=audio.dtype,
                channels=audio.shape[0],
                filters=self.filters,
                dtype=self.dtype,
                output_sample_rate=self.output_sample_rate,
                to_mono=self.to_mono,
                speed=self.speed,
                pitch_shift=self.pitch_shift,
                frame_size=self.frame_size,
            )
        self._processor.push(audio)
        self._buffered_bytes = buffered_bytes

    def pull(self, partial: bool = False) -> Iterator[tuple[np.ndarray, int]]:
        if self._processor is None:
            return
        completed = False
        try:
            for audio, sample_rate in self._processor.pull(partial=partial):
                if self.fill_value is not None:
                    audio = pad(audio, self.frame_size, self.fill_value)
                yield audio if self.always_2d else squeeze_mono(audio), sample_rate
            completed = True
        finally:
            if completed:
                self._buffered_bytes = self._processor.buffered_bytes
                self._finalized = partial

    @property
    def buffered_bytes(self) -> int:
        processor_bytes = 0 if self._processor is None else self._processor.buffered_bytes
        return max(self._buffered_bytes, processor_bytes)

    def close(self) -> None:
        if self._processor is not None:
            self._processor.close()
            self._processor = None
        self._buffered_bytes = 0
        self._finalized = True
        self._closed = True

    def reset(self) -> None:
        if self._processor is not None:
            self._processor.close()
            self._processor = None
        self._buffered_bytes = 0
        self._finalized = False
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
