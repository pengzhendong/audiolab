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
from importlib.resources import files
from io import BytesIO
from typing import Tuple, Union

import numpy as np
from av import AudioFormat, AudioFrame, AudioLayout
from av.audio.frame import format_dtypes
from jinja2 import Environment, FileSystemLoader
from lhotse import Seconds
from lhotse.caching import AudioCache
from lhotse.utils import SmartOpen

format_dtypes = {**format_dtypes, "s64": "i8", "s64p": "i8"}
dtypes = {np.dtype(v): k for k, v in format_dtypes.items() if "p" not in k}
loader = FileSystemLoader(files("audiolab.pyav").joinpath("templates"))
logger = logging.getLogger(__name__)


def get_template(name: str) -> str:
    return Environment(loader=loader).get_template(f"{name}.txt")


def load_url(url: str) -> BytesIO:
    audio_bytes = AudioCache.try_cache(url)
    if not audio_bytes:
        with SmartOpen.open(url, "rb") as f:
            audio_bytes = f.read()
        AudioCache.add_to_cache(url, audio_bytes)
    return BytesIO(audio_bytes)


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
            if np.dtype(dtype).kind in ("i", "u"):
                ndarray = ndarray * np.iinfo(dtype).max
            ndarray = ndarray.astype(dtype)
        else:
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


def split_audio_frame(frame: AudioFrame, offset: Seconds) -> Tuple[AudioFrame, AudioFrame]:
    offset = int(offset * frame.rate)
    if offset <= 0:
        return frame, None
    # Number of audio samples (per channel).
    if offset > frame.samples:
        return None, frame

    ndarray = to_ndarray(frame)
    left, right = ndarray[:, :offset], ndarray[:, offset:]
    if frame.format.is_packed:
        left, right = left.T.reshape(1, -1), right.T.reshape(1, -1)
    left = AudioFrame.from_ndarray(left, format=frame.format.name, layout=frame.layout)
    right = AudioFrame.from_ndarray(right, format=frame.format.name, layout=frame.layout)
    left.pts, right.pts = frame.pts, frame.pts + offset
    left.rate, right.rate = frame.rate, frame.rate
    left.time_base, right.time_base = frame.time_base, frame.time_base
    return left, right
