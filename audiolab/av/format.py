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

import logging
from functools import lru_cache
from typing import Dict, Iterable, Literal, Optional, Set, Union

import av
import numpy as np
from av import Codec, codecs_available
from av.codec.codec import UnknownCodecError

from audiolab.av import typing
from audiolab.av.utils import get_template

logger = logging.getLogger(__name__)
format_dtypes = {
    "dbl": "f8",
    "dblp": "f8",
    "flt": "f4",
    "fltp": "f4",
    "s16": "i2",
    "s16p": "i2",
    "s32": "i4",
    "s32p": "i4",
    "s64": "i8",
    "s64p": "i8",
    "u8": "u1",
    "u8p": "u1",
}
dtype_formats = {np.dtype(dtype): name for name, dtype in format_dtypes.items() if not name.endswith("p")}
audio_formats: Dict[str, av.AudioFormat] = {name: av.AudioFormat(name) for name in format_dtypes.keys()}
AudioFormat = typing.AudioFormatEnum("AudioFormat", audio_formats)


@lru_cache(maxsize=None)
def get_codecs(format: typing.AudioFormat, mode: Literal["r", "w"] = "r") -> Set[str]:
    """
    Get the codecs available for an audio format.

    Args:
        format: The audio format.
        mode: The mode to get the codecs.
    Returns:
        The codecs available for the audio format.
    """
    codecs = set()
    if isinstance(format, av.AudioFormat):
        format = format.name
    for codec in codecs_available:
        try:
            codec = Codec(codec, mode)
            formats = codec.audio_formats
            if codec.type != "audio" or formats is None:
                continue
            if format in set(format.name for format in formats):
                codecs.add(codec.name)
        except UnknownCodecError:
            pass
    return codecs


@lru_cache(maxsize=None)
def get_format_dtype(format: typing.AudioFormat) -> np.dtype:
    """
    Get the data type of an audio format.

    Args:
        format: The audio format.
    Returns:
        The data type of the audio format.
    """
    if isinstance(format, av.AudioFormat):
        format = format.name
    return np.dtype(format_dtypes[format])


def get_format(
    dtype: Union[str, type, np.dtype],
    is_planar: Optional[bool] = None,
    available_formats: Optional[Iterable[typing.AudioFormat]] = None,
) -> av.AudioFormat:
    """
    Get the audio format of an audio data type.

    Args:
        dtype: The type of the audio data, such as "float32", or float.
        is_planar: Whether the audio is planar.
        available_formats: The available formats.
    Returns:
        The audio format of the audio data type.
    """
    if isinstance(dtype, str) and dtype not in format_dtypes or isinstance(dtype, type):
        dtype = np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        assert dtype in dtype_formats, f"Input dtype `{dtype}` is not in {dtype_formats}."
        dtype = dtype_formats[dtype]
        if is_planar is not None:
            dtype = dtype + ("p" if is_planar else "")
        else:
            assert available_formats is not None
            available_formats = [
                format.name if isinstance(format, typing.AudioFormat) else format for format in available_formats
            ]
            if dtype not in available_formats:
                opposite_format = "packed" if dtype.endswith("p") else "planar"
                logger.warning(f"Input format `{dtype}` is not in {available_formats}, try {opposite_format} format.")
                dtype = dtype.rstrip("p") if dtype.endswith("p") else dtype + "p"
            assert dtype in available_formats, f"Input format `{dtype}` is not in {available_formats}."
    return AudioFormat[dtype].value


template = get_template("format")
for name, format in audio_formats.items():
    decodecs = get_codecs(name, "r")
    encodecs = get_codecs(name, "w")
    dtype = get_format_dtype(name)
    getattr(AudioFormat, name).__doc__ = template.render(
        format=format, decodecs=decodecs, encodecs=encodecs, dtype=dtype
    )
