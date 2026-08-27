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

from fractions import Fraction

import av
import numpy as np
from numpy.typing import DTypeLike

from audiolab.av.format import get_dtype
from audiolab.av.typing import AudioFormatLike, AudioLayoutLike


def _integer_scale(dtype: np.dtype) -> int:
    limits = np.iinfo(dtype)
    return limits.max if dtype.kind == "u" else limits.max + 1


def clip(audio: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    source_dtype = audio.dtype
    target_dtype = np.dtype(dtype)
    if source_dtype.kind not in "fiu" or target_dtype.kind not in "fiu":
        raise TypeError("Audio conversion requires numeric integer or floating-point dtypes")
    if audio.size == 0:
        return audio.astype(target_dtype, copy=False)
    if source_dtype.kind != "f" and source_dtype == target_dtype:
        return audio

    if source_dtype.kind == "f":
        converted = np.clip(audio, -1.0, 1.0)
        if target_dtype.kind == "f":
            return converted.astype(target_dtype, copy=False)
        source_weight = 1.0
        source_bias = 0.0
    else:
        work_dtype = target_dtype if target_dtype.kind == "f" else np.dtype(np.float64)
        converted = audio.astype(work_dtype)
        if source_dtype.kind == "u":
            source_weight = 2.0 / _integer_scale(source_dtype)
            source_bias = -1.0
        else:
            source_weight = 1.0 / _integer_scale(source_dtype)
            source_bias = 0.0

    if target_dtype.kind == "f":
        target_weight = 1.0
        target_bias = 0.0
    elif target_dtype.kind == "u":
        target_weight = 0.5 * _integer_scale(target_dtype)
        target_bias = target_weight
    else:
        target_weight = _integer_scale(target_dtype)
        target_bias = 0.0

    weight = source_weight * target_weight
    bias = source_bias * target_weight + target_bias
    if weight != 1.0:
        np.multiply(converted, weight, out=converted)
    if bias != 0.0:
        np.add(converted, bias, out=converted)
    if target_dtype.kind in "iu":
        limits = np.iinfo(target_dtype)
        np.clip(converted, limits.min, limits.max, out=converted)
    return converted.astype(target_dtype)


def squeeze_mono(audio: np.ndarray) -> np.ndarray:
    """Remove only a singleton channel axis, preserving the sample axis."""
    if audio.ndim == 2 and audio.shape[0] == 1:
        return audio[0]
    return audio


def from_ndarray(
    audio: np.ndarray,
    sample_format: AudioFormatLike,
    layout: AudioLayoutLike,
    sample_rate: int,
    pts: int | None = None,
    time_base: Fraction | None = None,
) -> av.AudioFrame:
    audio = np.atleast_2d(audio)
    if isinstance(sample_format, str):
        sample_format = av.AudioFormat(sample_format)
    if sample_format.is_packed:
        # [num_channels, num_samples] => [1, num_channels * num_samples]
        audio = audio.T.reshape(1, -1)
    if isinstance(layout, str):
        layout = av.AudioLayout(layout)

    dtype = get_dtype(sample_format)
    audio = np.ascontiguousarray(clip(audio, dtype))
    audio_frame = av.AudioFrame.from_ndarray(audio, sample_format.name, layout)
    audio_frame.rate = sample_rate
    if pts is not None:
        audio_frame.pts = pts
    if time_base is not None:
        audio_frame.time_base = time_base
    return audio_frame


def to_ndarray(audio_frame: av.AudioFrame) -> np.ndarray:
    # packed: [num_channels, num_samples]
    # planar: [1, num_channels * num_samples]
    audio = audio_frame.to_ndarray()
    if audio_frame.format.is_packed:
        audio = audio.reshape(-1, audio_frame.layout.nb_channels).T
    return audio


def split_audio_frame(audio_frame: av.AudioFrame, offset: int) -> tuple[av.AudioFrame | None, av.AudioFrame | None]:
    if offset <= 0:
        return None, audio_frame
    # number of samples per channel
    if offset >= audio_frame.samples:
        return audio_frame, None

    audio = to_ndarray(audio_frame)
    left, right = audio[:, :offset], audio[:, offset:]
    if audio_frame.format.is_packed:
        left, right = left.T.reshape(1, -1), right.T.reshape(1, -1)
    left = av.AudioFrame.from_ndarray(left, audio_frame.format.name, audio_frame.layout)
    right = av.AudioFrame.from_ndarray(right, audio_frame.format.name, audio_frame.layout)
    left.rate, right.rate = audio_frame.rate, audio_frame.rate
    if audio_frame.pts is not None:
        left.pts, right.pts = audio_frame.pts, audio_frame.pts + offset
    if audio_frame.time_base is not None:
        left.time_base, right.time_base = audio_frame.time_base, audio_frame.time_base
    return left, right


def pad(audio: np.ndarray, frame_size: int, fill_value: float = 0) -> np.ndarray:
    pad_needed = frame_size - audio.shape[0 if audio.ndim == 1 else 1]
    if pad_needed <= 0:
        return audio
    if audio.ndim == 1:
        return np.pad(audio, (0, pad_needed), constant_values=fill_value)
    return np.pad(audio, ((0, 0), (0, pad_needed)), constant_values=fill_value)
