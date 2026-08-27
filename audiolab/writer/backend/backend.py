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
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from audiolab.av.frame import clip
from audiolab.av.typing import ContainerFormatLike


class Backend:
    def __init__(
        self,
        destination: Any,
        sample_rate: int,
        dtype: DTypeLike | None = None,
        container_format: ContainerFormatLike = "WAV",
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.destination = destination
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.container_format = container_format
        self.num_channels = None

    def prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim != 2:
            raise ValueError("audio must have shape (samples,) or (channels, samples)")
        if audio.shape[0] == 0:
            raise ValueError("audio must contain at least one channel")
        if self.dtype is None:
            self.dtype = audio.dtype
        audio = clip(audio, self.dtype)
        if self.num_channels is None:
            self.num_channels = audio.shape[0]
        elif audio.shape[0] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, received {audio.shape[0]}")
        return audio

    def close(self):
        destination = self.destination
        if destination is None:
            return
        self.destination = None
        if isinstance(destination, BytesIO):
            with contextlib.suppress(AttributeError, ValueError):
                destination.seek(0)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
