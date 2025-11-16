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

import io
import os
from functools import cached_property
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
from av.codec import Codec

from audiolab.av.typing import Seconds


class Backend:
    def __init__(self, file: Any, forced_encoding: bool = False):
        self.file = file
        self.forced_encoding = forced_encoding

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
    def format(self):
        pass

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
        return True

    @cached_property
    def size(self) -> Optional[int]:
        if isinstance(self.file, str):
            if os.path.exists(self.file):
                return os.stat(self.file).st_size
        elif isinstance(self.file, BytesIO):
            return len(self.file.getbuffer())
        return None

    def read(self, frames: int = np.iinfo(np.int32).max) -> np.ndarray:
        pass

    def seek(self, offset: Seconds, whence: int = io.SEEK_SET) -> int:
        pass
