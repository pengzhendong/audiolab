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

_dtype_to_bytes_map = {"uint8": 1, "int16": 2, "int32": 4}


class Wave:
    def __init__(self, file: Any, num_channels: int, sample_rate: int, dtype: Optional[Dtype] = None):
        self.wave = wave.open(file, "w")
        self.wave.setnchannels(num_channels)
        self.wave.setframerate(sample_rate)
        if dtype is None:
            dtype = np.int16
        self.dtype = np.dtype(dtype)
        sampwidth = _dtype_to_bytes_map[self.dtype.name]
        self.wave.setsampwidth(sampwidth)

    def write(self, frame: np.ndarray):
        frame = clip(frame, self.dtype)
        # [num_channels, num_samples] => [num_samples, num_channels]
        self.wave.writeframes(frame.T.tobytes())

    def close(self):
        self.wave.close()
