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

import contextlib
import os
from collections.abc import Iterator
from functools import cached_property
from io import BytesIO
from typing import Any

import numpy as np

from audiolab.av import standard_channel_layouts
from audiolab.av.typing import UINT32_MAX, Seconds

FORCED_DECODE_CHUNK_FRAMES = 65_536


class Backend:
    def __init__(self, source: Any, frame_size: int | None = None, forced_decoding: bool = False):
        if frame_size is not None and frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self.source = source
        self.frame_size = UINT32_MAX if frame_size is None else min(frame_size, UINT32_MAX)
        self.forced_decoding = forced_decoding

    def close(self):
        if self.source is None:
            return
        self.source = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    @cached_property
    def bit_rate(self) -> float | int | None:
        bit_rate = None
        if self.size is not None and self.duration is not None and self.duration > 0:
            bit_rate = self.size * 8 / self.duration
        return bit_rate

    @cached_property
    def is_planar(self) -> bool:
        return False

    @cached_property
    def layout(self) -> str:
        layouts = standard_channel_layouts[self.num_channels]
        return layouts[0]

    @cached_property
    def metadata(self) -> dict:
        return {}

    @cached_property
    def name(self) -> str:
        return "<none>" if isinstance(self.source, BytesIO) else self.source

    @cached_property
    def size(self) -> int | None:
        if isinstance(self.source, str):
            if os.path.exists(self.source):
                return os.stat(self.source).st_size
        elif isinstance(self.source, BytesIO):
            return len(self.source.getbuffer())
        return None

    def load_audio(self, offset: Seconds = 0, duration: Seconds | None = None) -> Iterator[np.ndarray]:
        self.seek(int(offset * self.sample_rate))
        remaining_frames = UINT32_MAX if duration is None else int(duration * self.sample_rate)
        while remaining_frames > 0:
            frame_size = min(remaining_frames, self.frame_size)
            audio = self.read(frame_size)
            if audio is None:
                break
            remaining_frames -= audio.shape[1]
            yield audio
