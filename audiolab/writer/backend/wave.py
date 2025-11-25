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

import wave
from typing import Any, Optional

import numpy as np

from audiolab.av.frame import clip
from audiolab.av.typing import Dtype
from audiolab.writer.backend.backend import Backend

_dtype_to_bytes = {"uint8": 1, "int16": 2, "int32": 4}


class Wave(Backend):
    def __init__(self, file: Any, sample_rate: int, dtype: Optional[Dtype] = None):
        super().__init__(file, sample_rate, dtype)
        self.wave = None
        self.num_channels = None

    def open(self):
        self.wave = wave.open(self.file, "w")
        self.wave.setframerate(self.sample_rate)
        self.wave.setnchannels(self.num_channels)
        sampwidth = _dtype_to_bytes[self.dtype.name]
        self.wave.setsampwidth(sampwidth)

    def write(self, frame: np.ndarray):
        if self.dtype is None:
            self.dtype = frame.dtype
        frame = np.atleast_2d(clip(frame, self.dtype))
        if self.num_channels is None:
            self.num_channels = frame.shape[0]
        if self.wave is None:
            self.open()
        self.wave.writeframes(frame.tobytes())

    def close(self):
        if self.wave is not None and not self.is_closed:
            self.wave.close()
            super().close()
