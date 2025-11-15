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

from functools import lru_cache
from typing import Dict, Iterable, Literal, Optional, Set, Union

import av
import numpy as np
from av import Codec, codecs_available
from av.codec.codec import UnknownCodecError

from audiolab.av import typing
from audiolab.av.utils import get_template


"""
$ ffmpeg -sample_fmts
"""
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
dtype_formats = {
    np.dtype(dtype): name
    for name, dtype in format_dtypes.items()
    if not name.endswith("p")
}
audio_formats: Dict[str, av.AudioFormat] = {
    name: av.AudioFormat(name) for name in format_dtypes.keys()
}
AudioFormat = typing.AudioFormatEnum("AudioFormat", audio_formats)


@lru_cache(maxsize=None)
def get_codecs(format: typing.AudioFormat, mode: Literal["r", "w"] = "r") -> Set[str]:
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
def get_dtype(format: typing.AudioFormat) -> np.dtype:
    if isinstance(format, av.AudioFormat):
        format = format.name
    return np.dtype(format_dtypes[format])


def get_format(
    dtype: Union[str, type, np.dtype],
    is_planar: Optional[bool] = None,
    available_formats: Optional[Iterable[typing.AudioFormat]] = None,
) -> av.AudioFormat:
    if isinstance(dtype, str) and dtype not in format_dtypes or isinstance(dtype, type):
        dtype = np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        dtype = dtype_formats[dtype]
        if is_planar is not None:
            dtype = dtype + ("p" if is_planar else "")
        else:
            assert available_formats is not None
            available_formats = [
                format.name if isinstance(format, typing.AudioFormat) else format
                for format in available_formats
            ]
            if dtype not in available_formats:
                dtype = dtype.rstrip("p") if dtype.endswith("p") else dtype + "p"
    return AudioFormat[dtype].value


template = get_template("format")
for name, format in audio_formats.items():
    decodecs = get_codecs(name, "r")
    encodecs = get_codecs(name, "w")
    dtype = get_dtype(name)
    getattr(AudioFormat, name).__doc__ = template.render(
        format=format, decodecs=decodecs, encodecs=encodecs, dtype=dtype
    )
