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

from audiolab.av.format import get_dtype
from audiolab.av.typing import AudioFormat, AudioLayout
from audiolab.av.utils import get_logger

logger = get_logger(__name__)


def clip(ndarray: np.ndarray, dtype: np.dtype) -> np.ndarray:
    source_type = ndarray.dtype
    target_type = np.dtype(dtype)
    if source_type.kind != "f" and source_type == target_type:
        return ndarray

    if source_type.kind == "f":
        min_value, max_value = ndarray.min(), ndarray.max()
        if min_value < -1.0 or max_value >= 1.0:
            logger.warning("Cliping %s ndarray from: %g ~ %g to -1.0 ~ 1.0", source_type, min_value, max_value)
            ndarray = np.clip(ndarray, -1.0, 1.0)
    else:
        ndarray = ndarray.astype(np.float64)
        if source_type.kind == "u":
            ndarray = ndarray / np.iinfo(source_type).max * 2 - 1
        elif source_type.kind == "i":
            ndarray = ndarray / (np.iinfo(source_type).max + 1)

    if target_type.kind == "u":
        ndarray = ((ndarray + 1) * 0.5 * np.iinfo(target_type).max).round()
    elif target_type.kind == "i":
        ndarray = (ndarray * np.iinfo(target_type).max).round()
    return np.asarray(ndarray, dtype=dtype)


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
    frame = av.AudioFrame.from_ndarray(ndarray, format=format.name, layout=layout)
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
    left = av.AudioFrame.from_ndarray(left, format=frame.format.name, layout=frame.layout)
    right = av.AudioFrame.from_ndarray(right, format=frame.format.name, layout=frame.layout)
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
