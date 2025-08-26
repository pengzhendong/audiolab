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
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import av
import numpy as np

from audiolab.av import (
    AudioCache,
    aformat,
    clip,
    from_ndarray,
    get_format,
    get_format_dtype,
    split_audio_frame,
    to_ndarray,
)
from audiolab.av.typing import AudioFormat, ContainerFormat, Dtype
from audiolab.reader import Reader, StreamReader, info, load_audio
from audiolab.writer import Writer, save_audio


def encode(
    audio: Union[str, Path, np.ndarray],
    rate: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    is_planar: bool = False,
    format: Optional[AudioFormat] = None,
    to_mono: bool = False,
    make_wav: bool = True,
    container_format: ContainerFormat = "wav",
) -> Tuple[str, int]:
    """
    Transform an audio to a PCM bytestring.

    Args:
        audio: The file path to an audio file or a numpy array containing raw audio samples.
        rate: The sample rate of the audio.
        dtype: The data type of the audio.
        is_planar: Whether the audio is planar.
        format: The format of the audio.
        to_mono: Whether to convert the audio to mono.
        make_wav: Whether to make the audio a WAV file.
        container_format: The format of the audio container.
    Returns:
        The audio as a PCM bytestring and the sample rate of the audio.
    """
    if isinstance(audio, (str, Path)):
        audio, rate = load_audio(audio, dtype=dtype, is_planar=is_planar, format=format, rate=rate)

    audio = clip(audio, np.int16)
    if to_mono and audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True).astype(audio.dtype)
    if make_wav:
        assert rate is not None
        bytestream = BytesIO()
        if isinstance(container_format, av.ContainerFormat):
            container_format = container_format.name
        save_audio(bytestream, audio, rate, container_format=container_format)
        audio = b64encode(bytestream.getvalue()).decode("ascii")
        audio = f"data:audio/{container_format};base64,{audio}"
    else:
        audio = np.ascontiguousarray(audio)
        audio = b64encode(audio).decode("ascii")
    return audio, rate


__all__ = [
    "AudioCache",
    "Reader",
    "StreamReader",
    "Writer",
    "aformat",
    "encode",
    "from_ndarray",
    "get_format",
    "get_format_dtype",
    "info",
    "load_audio",
    "save_audio",
    "split_audio_frame",
    "to_ndarray",
]
