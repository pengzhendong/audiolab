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
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import DTypeLike

from audiolab.writer.backend import pyav, soundfile


class Writer:
    """Incrementally write channels-first NumPy PCM chunks."""

    def __init__(
        self,
        destination: Any,
        sample_rate: int,
        dtype: DTypeLike | None = None,
        container_format: str = "WAV",
    ):
        backend = soundfile if container_format.upper() in sf.available_formats() else pyav
        self.backend = backend(destination, sample_rate, dtype, container_format)

    def write(self, audio: np.ndarray) -> None:
        self.backend.write(audio)

    def close(self):
        backend = self.backend
        if backend is not None:
            self.backend = None
            backend.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
