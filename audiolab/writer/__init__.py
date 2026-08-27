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

from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from audiolab.writer.writer import Writer


def save_audio(
    destination: Any,
    audio: np.ndarray,
    sample_rate: int,
    dtype: DTypeLike | None = None,
    container_format: str = "WAV",
) -> None:
    """Write a complete channels-first NumPy audio array to a destination."""
    with Writer(destination, sample_rate, dtype, container_format) as writer:
        writer.write(audio)


__all__ = ["Writer", "save_audio"]
