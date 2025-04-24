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

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import bv
import numpy as np
from lhotse import Recording
from lhotse.cut import Cut

from audiolab.av import aformat, clip, from_ndarray, get_format, get_format_dtype, info, split_audio_frame, to_ndarray
from audiolab.av.typing import AudioFormat, ContainerFormat, Dtype
from audiolab.reader import Reader, StreamReader, load_audio
from audiolab.writer import Writer, save_audio


def encode(
    audio: Union[str, Path, np.ndarray, Cut, Recording],
    rate: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    is_planar: bool = False,
    format: Optional[AudioFormat] = None,
    to_mono: bool = False,
    make_wav: bool = True,
    container_format: ContainerFormat = "wav",
) -> Tuple[str, int]:
    """Transform an audio to a PCM bytestring"""
    if isinstance(audio, (str, Path)):
        audio, rate = load_audio(audio, dtype=dtype, is_planar=is_planar, format=format, rate=rate, to_mono=to_mono)
    elif isinstance(audio, (Cut, Recording)):
        if rate is not None:
            audio = audio.resample(rate)
        rate = audio.sampling_rate
        audio = audio.load_audio()
    audio = clip(audio, np.int16)
    if make_wav:
        assert rate is not None
        bytestream = BytesIO()
        if isinstance(container_format, bv.ContainerFormat):
            container_format = container_format.name
        save_audio(bytestream, audio, rate, container_format=container_format)
        audio = b64encode(bytestream.getvalue()).decode("ascii")
        audio = f"data:audio/{container_format};base64,{audio}"
    else:
        audio = np.ascontiguousarray(audio)
        audio = b64encode(audio).decode("ascii")
    return audio, rate


__all__ = [
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
