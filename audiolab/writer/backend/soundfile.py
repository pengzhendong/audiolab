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
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import DTypeLike

from audiolab.writer.backend.backend import Backend

_dtype_to_subtype = {"int16": "PCM_16", "int32": "PCM_32", "float32": "FLOAT", "float64": "DOUBLE"}


class SoundFile(Backend):
    def __init__(
        self, destination: Any, sample_rate: int, dtype: DTypeLike | None = None, container_format: str = "WAV"
    ):
        super().__init__(destination, sample_rate, dtype, container_format)
        self.sf = None

    @cached_property
    def subtype(self) -> str:
        if self.dtype is None:
            subtype = sf.default_subtype(self.container_format)
            if subtype is None:
                raise ValueError(f"{self.container_format} has no default audio subtype")
            return subtype
        subtype = _dtype_to_subtype.get(self.dtype.name)
        if subtype is None:
            raise ValueError(f"Unsupported output dtype: {self.dtype.name}")
        if not sf.check_format(self.container_format, subtype):
            raise ValueError(f"{self.container_format} does not support subtype {subtype}")
        return subtype

    def open(self):
        self.sf = sf.SoundFile(
            self.destination,
            "w",
            self.sample_rate,
            self.num_channels,
            self.subtype,
            format=self.container_format,
        )

    def write(self, audio: np.ndarray):
        audio = self.prepare_audio(audio)
        if self.sf is None:
            self.open()
        # (num_channels, num_samples) => (num_samples, num_channels)
        self.sf.write(audio.T)

    def close(self):
        sound_file = self.sf
        self.sf = None
        try:
            if sound_file is not None:
                sound_file.close()
        finally:
            super().close()
