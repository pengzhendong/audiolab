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

import os
from functools import cached_property
from io import BytesIO
from typing import Any, Iterator, Optional

import numpy as np

from audiolab.av import standard_channel_layouts
from audiolab.av.typing import UINT32_MAX, Seconds


class Backend:
    def __init__(self, file: Any, frame_size: Optional[int] = None, forced_decoding: bool = False):
        self.file = file
        if frame_size is None:
            frame_size = UINT32_MAX
        self.frame_size = min(frame_size, UINT32_MAX)
        self.forced_decoding = forced_decoding

    @cached_property
    def bit_rate(self) -> Optional[int]:
        bit_rate = None
        if self.size is not None:
            if self.duration is not None and self.duration > 0:
                bit_rate = self.size * 8 / self.duration
        return bit_rate

    @cached_property
    def layout(self) -> str:
        layouts = standard_channel_layouts[self.num_channels]
        return layouts[0]

    @cached_property
    def metadata(self) -> dict:
        return {}

    @cached_property
    def name(self) -> str:
        return "<none>" if isinstance(self.file, BytesIO) else self.file

    @cached_property
    def size(self) -> Optional[int]:
        if isinstance(self.file, str):
            if os.path.exists(self.file):
                return os.stat(self.file).st_size
        elif isinstance(self.file, BytesIO):
            return len(self.file.getbuffer())
        return None

    def load_audio(self, offset: Seconds = 0, duration: Optional[Seconds] = None) -> Iterator[np.ndarray]:
        self.seek(int(offset * self.sample_rate))
        frames = UINT32_MAX if duration is None else int(duration * self.sample_rate)
        while frames > 0:
            frame_size = min(frames, self.frame_size)
            ndarray = self.read(frame_size)
            if ndarray is None:
                break
            frames -= ndarray.shape[1]
            yield ndarray
