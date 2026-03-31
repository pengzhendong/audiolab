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
from typing import Optional, Tuple

import av
import numpy as np
from numpy.typing import DTypeLike

from audiolab.av.format import get_dtype
from audiolab.av.typing import AudioFormat, AudioLayout

_IINFO_CACHE = {np.dtype("int8"): 128, np.dtype("uint8"): 255, np.dtype("int16"): 32768, np.dtype("int32"): 2147483648}


def clip(ndarray: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    if ndarray.size == 0:
        return ndarray
    src_dtype = ndarray.dtype
    dst_dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
    if src_dtype.kind != "f" and src_dtype == dst_dtype:
        return ndarray

    if src_dtype.kind == "f":
        ndarray = np.clip(ndarray, -1.0, 1.0)
        src_weight = 1.0
        src_bias = 0.0
    else:
        ndarray = ndarray.astype(np.float64)
        if src_dtype.kind == "u":
            src_weight = 1.0 / _IINFO_CACHE[src_dtype] * 2
            src_bias = -1.0
        elif src_dtype.kind == "i":
            src_weight = 1.0 / _IINFO_CACHE[src_dtype]
            src_bias = 0.0

    if dst_dtype.kind == "f":
        dst_weight = 1.0
        dst_bias = 0.0
    elif dst_dtype.kind == "u":
        dst_weight = 0.5 * _IINFO_CACHE[dst_dtype]
        dst_bias = dst_weight
    elif dst_dtype.kind == "i":
        dst_weight = _IINFO_CACHE[dst_dtype]
        dst_bias = 0.0

    weight = src_weight * dst_weight
    bias = src_bias * dst_weight + dst_bias
    if weight != 1.0:
        np.multiply(ndarray, weight, out=ndarray)
    if bias != 0.0:
        np.add(ndarray, bias, out=ndarray)
    return ndarray.astype(dst_dtype)


def from_ndarray(
    ndarray: np.ndarray,
    format: AudioFormat,
    layout: AudioLayout,
    rate: int,
    pts: Optional[int] = None,
    time_base: Optional[Fraction] = None,
) -> av.AudioFrame:
    ndarray = np.atleast_2d(ndarray)
    if isinstance(format, str):
        format = av.AudioFormat(format)
    if format.is_packed:
        # [num_channels, num_samples] => [1, num_channels * num_samples]
        ndarray = ndarray.T.reshape(1, -1)
    if isinstance(layout, str):
        layout = av.AudioLayout(layout)

    dtype = get_dtype(format)
    ndarray = clip(ndarray, dtype)
    ndarray = np.ascontiguousarray(ndarray)
    frame = av.AudioFrame.from_ndarray(ndarray, format.name, layout)
    frame.rate = rate
    if pts is not None:
        frame.pts = pts
    if time_base is not None:
        frame.time_base = time_base
    return frame


def to_ndarray(frame: av.AudioFrame) -> np.ndarray:
    # packed: [num_channels, num_samples]
    # planar: [1, num_channels * num_samples]
    ndarray = frame.to_ndarray()
    if frame.format.is_packed:
        ndarray = ndarray.reshape(-1, frame.layout.nb_channels).T
    return ndarray


def split_audio_frame(frame: av.AudioFrame, offset: int) -> Tuple[av.AudioFrame, av.AudioFrame]:
    if offset <= 0:
        return None, frame
    # number of samples per channel
    if offset >= frame.samples:
        return frame, None

    ndarray = to_ndarray(frame)
    left, right = ndarray[:, :offset], ndarray[:, offset:]
    if frame.format.is_packed:
        left, right = left.T.reshape(1, -1), right.T.reshape(1, -1)
    left = av.AudioFrame.from_ndarray(left, frame.format.name, frame.layout)
    right = av.AudioFrame.from_ndarray(right, frame.format.name, frame.layout)
    left.rate, right.rate = frame.rate, frame.rate
    if frame.pts is not None:
        left.pts, right.pts = frame.pts, frame.pts + offset
    if frame.time_base is not None:
        left.time_base, right.time_base = frame.time_base, frame.time_base
    return left, right


def pad(frame: np.ndarray, frame_size: int, fill_value: float = 0) -> np.ndarray:
    pad_needed = frame_size - frame.shape[0 if frame.ndim == 1 else 1]
    if pad_needed <= 0:
        return frame
    if frame.ndim == 1:
        return np.pad(frame, (0, pad_needed), constant_values=fill_value)
    else:
        return np.pad(frame, ((0, 0), (0, pad_needed)), constant_values=fill_value)
