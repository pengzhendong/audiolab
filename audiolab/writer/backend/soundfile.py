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

from functools import cached_property
from typing import Any, Optional

import numpy as np
import soundfile as sf

from audiolab.av.frame import clip
from audiolab.av.typing import Dtype
from audiolab.writer.backend.backend import Backend

_dtype_to_subtype = {"int16": "PCM_16", "int32": "PCM_32", "float32": "FLOAT", "float64": "DOUBLE"}


class SoundFile(Backend):
    def __init__(
        self,
        file: Any,
        sample_rate: int,
        dtype: Optional[Dtype] = None,
        format: str = "WAV",
    ):
        super().__init__(file, sample_rate, dtype, format)
        self.sf = None
        self.num_channels = None

    @cached_property
    def subtype(self) -> str:
        if self.dtype is None:
            return sf.default_subtype(self.format)
        subtype = _dtype_to_subtype[self.dtype.name]
        # assert subtype in sf.available_subtypes(self.format)
        assert sf.check_format(self.format, subtype)
        return subtype

    def open(self):
        self.sf = sf.SoundFile(self.file, "w", self.sample_rate, self.num_channels, self.subtype, format=self.format)

    def write(self, frame: np.ndarray):
        if self.dtype is None:
            self.dtype = frame.dtype
        frame = np.atleast_2d(clip(frame, self.dtype))
        if self.num_channels is None:
            self.num_channels = frame.shape[0]
        if self.sf is None:
            self.open()
        # (num_channels, num_samples) => (num_samples, num_channels)
        self.sf.write(frame.T)

    def close(self):
        if self.sf is not None and not self.is_closed:
            self.sf.close()
            super().close()
