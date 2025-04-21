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

import logging
from fractions import Fraction
from typing import Optional, Tuple

import bv
import numpy as np
from lhotse import Seconds

from audiolab.av.format import get_format_dtype
from audiolab.av.typing import AudioFormat, AudioLayout

logger = logging.getLogger(__name__)


def clip(ndarray: np.ndarray, dtype: np.dtype) -> np.ndarray:
    ndarray = ndarray.numpy() if "tensor" in ndarray.__class__.__name__.lower() else ndarray
    source_type = ndarray.dtype
    target_type = np.dtype(dtype)
    assert source_type.kind in ("f", "i", "u") and target_type.kind in ("f", "i", "u")

    min_value = np.iinfo(source_type).min if source_type.kind in ("i", "u") else -1
    max_value = np.iinfo(source_type).max if source_type.kind in ("i", "u") else 1
    if np.any(ndarray < min_value) or np.any(ndarray > max_value):
        logger.warning(f"{source_type} array out of range: {ndarray.min()} ~ {ndarray.max()}")
        ndarray = np.clip(ndarray, min_value, max_value)

    ndarray = ndarray.astype(np.float64)
    if source_type.kind == "u":
        ndarray = ndarray / np.iinfo(source_type).max * 2 - 1
    elif source_type.kind == "i":
        ndarray = ndarray / np.iinfo(source_type).max

    if target_type.kind == "u":
        ndarray = ((ndarray + 1) * 0.5 * np.iinfo(target_type).max).round()
    elif target_type.kind == "i":
        ndarray = (ndarray * np.iinfo(target_type).max).round()
    return ndarray.astype(dtype)


def from_ndarray(
    ndarray: np.ndarray,
    format: AudioFormat,
    layout: AudioLayout,
    rate: int,
    pts: Optional[int] = None,
    time_base: Optional[Fraction] = None,
) -> bv.AudioFrame:
    if isinstance(format, str):
        format = bv.AudioFormat(format)
    if format.is_packed:
        # [num_channels, num_samples] => [1, num_channels * num_samples]
        ndarray = ndarray.T.reshape(1, -1)
    if isinstance(layout, str):
        layout = bv.AudioLayout(layout)

    dtype = get_format_dtype(format)
    ndarray = clip(ndarray, dtype)
    frame = bv.AudioFrame.from_ndarray(ndarray, format=format.name, layout=layout)
    frame.rate = rate
    if pts is not None:
        frame.pts = pts
    if time_base is not None:
        frame.time_base = time_base
    return frame


def to_ndarray(frame: bv.AudioFrame) -> np.ndarray:
    # packed: [num_channels, num_samples]
    # planar: [1, num_channels * num_samples]
    ndarray = frame.to_ndarray()
    if frame.format.is_packed:
        ndarray = ndarray.reshape(-1, frame.layout.nb_channels).T
    return ndarray  # [num_channels, num_samples]


def split_audio_frame(frame: bv.AudioFrame, offset: Seconds) -> Tuple[bv.AudioFrame, bv.AudioFrame]:
    offset = int(offset * frame.rate)
    if offset <= 0:
        return None, frame
    # Number of audio samples (per channel).
    if offset > frame.samples:
        return frame, None

    ndarray = to_ndarray(frame)
    left, right = ndarray[:, :offset], ndarray[:, offset:]
    if frame.format.is_packed:
        left, right = left.T.reshape(1, -1), right.T.reshape(1, -1)
    left = bv.AudioFrame.from_ndarray(left, format=frame.format.name, layout=frame.layout)
    right = bv.AudioFrame.from_ndarray(right, format=frame.format.name, layout=frame.layout)
    left.rate, right.rate = frame.rate, frame.rate
    if frame.pts is not None:
        left.pts, right.pts = frame.pts, frame.pts + offset
    if frame.time_base is not None:
        left.time_base, right.time_base = frame.time_base, frame.time_base
    return left, right
