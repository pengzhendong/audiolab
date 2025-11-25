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

from typing import Any, Optional

import numpy as np
import soundfile as sf

from audiolab.av.typing import Dtype
from audiolab.writer.backend import pyav, soundfile


class Writer:
    def __init__(self, file: Any, rate: int, dtype: Optional[Dtype] = None, format: str = "WAV"):
        backend = soundfile if format.upper() in sf.available_formats() else pyav
        self.backend = backend(file, rate, dtype, format)

    def write(self, frame: np.ndarray):
        self.backend.write(frame)

    def close(self):
        self.backend.close()
