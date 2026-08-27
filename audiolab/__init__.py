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

from __future__ import annotations

from base64 import b64encode
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike

from audiolab.av import (
    clip,
    from_ndarray,
    get_dtype,
    get_format,
    split_audio_frame,
    to_ndarray,
)
from audiolab.pipe import AudioPipe
from audiolab.reader import Reader, StreamReader, info, load_audio
from audiolab.writer import Writer, save_audio

try:
    __version__ = _distribution_version("audiolab")
except _PackageNotFoundError:
    __version__ = "0+unknown"


def encode(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    dtype: DTypeLike | None = None,
    to_mono: bool = False,
    include_container: bool = True,
    container_format: str = "WAV",
) -> tuple[str, int]:
    """
    Transform an audio to a PCM bytestring.

    Args:
        audio: The file path to an audio file or a numpy array.
        sample_rate: The sample rate of the audio.
        dtype: The data type of the audio.
        to_mono: Whether to convert the audio to mono.
        include_container: Encode a complete audio container instead of raw PCM bytes.
        container_format: The format of the audio container.
    Returns:
        The audio as a PCM bytestring and the sample rate of the audio.
    """
    if isinstance(audio, (str, Path)):
        audio, sample_rate = load_audio(audio, dtype=dtype, sample_rate=sample_rate, to_mono=to_mono)

    if sample_rate is None:
        raise ValueError("sample_rate is required when encoding a NumPy array")

    audio = clip(audio, np.int16)
    if include_container:
        bytestream = BytesIO()
        save_audio(bytestream, audio, sample_rate, container_format=container_format)
        audio = b64encode(bytestream.getvalue()).decode("ascii")
        audio = f"data:audio/{container_format};base64,{audio}"
    else:
        audio = np.ascontiguousarray(audio)
        audio = b64encode(audio).decode("ascii")
    return audio, sample_rate


__all__ = [
    "AudioPipe",
    "Reader",
    "StreamReader",
    "Writer",
    "__version__",
    "encode",
    "from_ndarray",
    "get_dtype",
    "get_format",
    "info",
    "load_audio",
    "save_audio",
    "split_audio_frame",
    "to_ndarray",
]
