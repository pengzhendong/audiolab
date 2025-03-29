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

from io import BytesIO
from typing import Tuple

import numpy as np
from av import AudioFrame
from lhotse import Seconds
from lhotse.caching import AudioCache
from lhotse.utils import SmartOpen


def load_url(url: str) -> BytesIO:
    audio_bytes = AudioCache.try_cache(url)
    if not audio_bytes:
        with SmartOpen.open(url, "rb") as f:
            audio_bytes = f.read()
        AudioCache.add_to_cache(url, audio_bytes)
    return BytesIO(audio_bytes)


def to_ndarray(frame: AudioFrame) -> np.ndarray:
    ndarray = frame.to_ndarray()
    if frame.format.is_packed:
        ndarray = ndarray.reshape(-1, frame.layout.nb_channels).T
    return ndarray


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
