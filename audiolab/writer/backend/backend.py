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
from typing import Any, Optional

import numpy as np

from audiolab.av.typing import Dtype


class Backend:
    def __init__(self, file: Any, sample_rate: int, dtype: Optional[Dtype] = None, format: str = "WAV"):
        self.file = file
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.format = format

    def close(self):
        file = self.file
        if file is None:
            return
        self.file = None
        if isinstance(file, BytesIO):
            try:
                file.seek(0)
            except (AttributeError, ValueError):
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
