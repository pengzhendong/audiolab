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

from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from soundfile import LibsndfileError

from audiolab.av.typing import FilterSpec, Seconds
from audiolab.reader.info import Info
from audiolab.reader.reader import DEFAULT_READ_FRAMES, Reader
from audiolab.reader.stream_reader import StreamReader


def info(source: Any, forced_decoding: bool = False, backends: list[str] | None = None) -> Info:
    """
    Get the information of an audio file.

    Args:
        source: The input audio file, audio URL, path, or encoded bytes.
        forced_decoding: Whether to forced decoding the audio file to get the duration.
        backends: The list of backends to use to get the information.
    Returns:
        The information of the audio file.
    """
    return Info(source, forced_decoding=forced_decoding, backends=backends)


def load_audio(
    source: Any,
    *,
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
) -> tuple[np.ndarray, int]:
    """Decode an entire audio source into memory.

    ``Reader`` is the streaming API. This convenience function is deliberately
    eager and always returns one audio array plus its sample rate, regardless
    of the reader's internal ``frame_size``.
    """
    options = {
        "offset": offset,
        "duration": duration,
        "filters": filters,
        "dtype": dtype,
        "sample_rate": sample_rate,
        "to_mono": to_mono,
        "speed": speed,
        "pitch_shift": pitch_shift,
        "frame_size": frame_size,
        "read_size": read_size,
        "cache_url": cache_url,
        "always_2d": always_2d,
        "fill_value": fill_value,
        "backends": backends,
    }
    chunks = []
    output_rate = None
    with Reader(source, **options) as reader:
        output = _allocate_eager_output(
            reader,
            offset=offset,
            duration=duration,
            filters=filters,
            sample_rate=sample_rate,
            to_mono=to_mono,
            speed=speed,
            always_2d=always_2d,
        )
        write_position = 0
        try:
            for chunk, chunk_rate in reader:
                if output_rate is not None and output_rate != chunk_rate:
                    raise RuntimeError("Audio sample rate changed while decoding")
                output_rate = chunk_rate
                chunk_length = chunk.shape[-1]
                if output is not None and write_position + chunk_length <= output.shape[-1]:
                    output[..., write_position : write_position + chunk_length] = chunk
                    write_position += chunk_length
                else:
                    if output is not None:
                        chunks.append(output[..., :write_position])
                        output = None
                    chunks.append(chunk)
        except LibsndfileError as error:
            if str(error) != "Internal psf_fseek() failed.":
                raise

        if output is not None and write_position > 0:
            if output_rate is None:
                raise RuntimeError("Decoded audio did not provide a sample rate")
            if write_position == output.shape[-1]:
                return output, output_rate
            return output[..., :write_position].copy(), output_rate

        if not chunks:
            if output is not None and output.shape[-1] == 0:
                return output, sample_rate or reader.sample_rate
            shape = (0, 0) if always_2d else (0,)
            return np.empty(shape, dtype=reader.output_dtype), reader.output_sample_rate

    axis = 1 if chunks[0].ndim == 2 else 0
    if output_rate is None:
        raise RuntimeError("Decoded audio did not provide a sample rate")
    if len(chunks) == 1:
        return chunks[0], output_rate
    return np.concatenate(chunks, axis=axis), output_rate


def _allocate_eager_output(
    reader: Reader,
    *,
    offset: Seconds,
    duration: Seconds | None,
    filters: list[FilterSpec] | None,
    sample_rate: int | None,
    to_mono: bool,
    speed: float,
    always_2d: bool,
) -> np.ndarray | None:
    """Preallocate predictable PCM output so eager loading does not retain every chunk."""
    backend = getattr(reader, "backend", None)
    num_frames = getattr(reader, "num_frames", None)
    if backend is None or filters or num_frames is None:
        return None

    source_rate = reader.sample_rate
    offset_frames = min(int(offset * source_rate), num_frames)
    num_frames -= offset_frames
    if duration is not None:
        num_frames = min(num_frames, int(duration * source_rate))
    output_rate = sample_rate or source_rate
    num_frames = round(num_frames * output_rate / source_rate / speed)

    num_channels = 1 if to_mono else reader.num_channels
    shape = (num_channels, num_frames) if always_2d or num_channels > 1 else (num_frames,)
    dtype = reader.output_dtype
    return np.empty(shape, dtype=dtype)


__all__ = ["Reader", "StreamReader", "load_audio"]
