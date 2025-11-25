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

import atexit
from io import BytesIO
from typing import Any, Optional

import numpy as np

from audiolab.av.typing import Dtype


class Backend:
    def __init__(self, file: Any, sample_rate: int, dtype: Optional[Dtype] = None, format: str = "WAV"):
        self.file = file
        self.sample_rate = sample_rate
        self.dtype = None
        if dtype is not None:
            self.dtype = np.dtype(dtype)
        self.format = format

        self.is_closed = False
        atexit.register(self.close)

    def close(self):
        if not self.is_closed:
            if isinstance(self.file, BytesIO):
                self.file.seek(0)
            self.is_closed = True
