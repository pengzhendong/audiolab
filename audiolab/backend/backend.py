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
from typing import Any, Iterator, Optional, Union

import numpy as np
from av.codec import Codec

from audiolab.av import load_url, standard_channel_layouts
from audiolab.av.typing import UINT32_MAX, Seconds
from audiolab.av.utils import pad


class Backend:
    def __init__(
        self,
        file: Any,
        frame_size: Optional[int] = None,
        frame_size_ms: Optional[int] = None,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
        cache_url: bool = False,
        forced_decoding: bool = False,
        **kwargs,
    ):
        if isinstance(file, str) and "://" in file and cache_url:
            file = load_url(file, cache=True)

        self.file = file
        self._frame_size = frame_size
        self.frame_size_ms = frame_size_ms
        self.always_2d = always_2d
        self.fill_value = fill_value
        self.forced_decoding = forced_decoding

    @cached_property
    def bits_per_sample(self) -> int:
        pass

    @cached_property
    def bit_rate(self) -> Optional[int]:
        bit_rate = None
        if self.size is not None:
            if self.duration is not None and self.duration > 0:
                bit_rate = self.size * 8 / self.duration
        return bit_rate

    @cached_property
    def codec(self) -> Union[Codec, str]:
        pass

    @cached_property
    def duration(self) -> Optional[Seconds]:
        pass

    @cached_property
    def dtype(self) -> np.dtype:
        pass

    @cached_property
    def format(self):
        pass

    @cached_property
    def frame_size(self) -> int:
        frame_size = self._frame_size
        if self.frame_size_ms is not None:
            frame_size = int(self.frame_size_ms * self.sample_rate // 1000)
        return min(UINT32_MAX if frame_size is None else frame_size, UINT32_MAX)

    @cached_property
    def layout(self) -> str:
        layouts = standard_channel_layouts[self.num_channels]
        return layouts[0]

    @cached_property
    def name(self) -> str:
        return "<none>" if isinstance(self.file, BytesIO) else self.file

    @cached_property
    def num_channels(self) -> int:
        pass

    @cached_property
    def num_frames(self) -> int:
        pass

    @cached_property
    def metadata(self) -> dict:
        return {}

    @cached_property
    def sample_rate(self) -> int:
        pass

    @cached_property
    def seekable(self) -> bool:
        pass

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
            if ndarray.shape[1] == 0:
                break
            frames -= frame_size

            if self.frame_size < UINT32_MAX and self.fill_value is not None:
                ndarray = pad(ndarray, self.frame_size, self.fill_value)
            if not self.always_2d and ndarray.shape[0] == 1:
                ndarray = ndarray[0]
            yield ndarray

    def read(self, nframes: int) -> np.ndarray:
        pass

    def seek(self, offset: Seconds):
        pass
