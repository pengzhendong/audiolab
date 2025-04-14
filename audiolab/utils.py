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
from typing import Union

import numpy as np
from av import AudioFormat, AudioFrame, AudioLayout
from av.audio.frame import format_dtypes

logger = logging.getLogger(__name__)


def from_ndarray(
    ndarray: np.ndarray, format: Union[str, AudioFormat], layout: Union[str, AudioLayout], rate: int
) -> AudioFrame:
    if isinstance(format, str):
        format = AudioFormat(format)
    if isinstance(layout, str):
        layout = AudioLayout(layout)
    if format.is_packed:
        # [num_channels, num_samples] => [1, num_channels * num_samples]
        ndarray = ndarray.T.reshape(1, -1)

    if ndarray.dtype.kind == "f" and (np.any(ndarray < -1) or np.any(ndarray >= 1)):
        logger.warning(f"Floating-point array out of range: {ndarray.min()} ~ {ndarray.max()}")
        ndarray = np.clip(ndarray, -1, 1)
    # Convert to the target dtype
    dtype = format_dtypes[format.name]
    if ndarray.dtype != dtype:
        if ndarray.dtype.kind in ("i", "u"):
            ndarray = ndarray / (np.iinfo(ndarray.dtype).max + 1)
        ndarray = (ndarray * np.iinfo(dtype).max).astype(dtype)

    frame = AudioFrame.from_ndarray(ndarray, format=format.name, layout=layout)
    frame.rate = rate
    return frame


def to_ndarray(frame: AudioFrame) -> np.ndarray:
    # packed: [num_channels, num_samples]
    # planar: [1, num_channels * num_samples]
    ndarray = frame.to_ndarray()
    if frame.format.is_packed:
        ndarray = ndarray.reshape(-1, frame.layout.nb_channels).T
    return ndarray  # [num_channels, num_samples]
