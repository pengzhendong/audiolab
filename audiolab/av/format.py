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

import bv
import numpy as np
from bv import Codec, codecs_available
from bv.audio.frame import format_dtypes
from bv.codec.codec import UnknownCodecError

from audiolab.av import typing
from audiolab.av.utils import get_template

logger = logging.getLogger(__name__)
format_dtypes = {**format_dtypes, "s64": "i8", "s64p": "i8"}
dtype_formats = {np.dtype(dtype): name for name, dtype in format_dtypes.items() if not name.endswith("p")}
audio_formats: Dict[str, bv.AudioFormat] = {name: bv.AudioFormat(name) for name in format_dtypes.keys()}
AudioFormat = typing.AudioFormatEnum("AudioFormat", audio_formats)


@lru_cache(maxsize=None)
def get_codecs(format: typing.AudioFormat, mode: Literal["r", "w"] = "r") -> Set[str]:
    codecs = set()
    if isinstance(format, bv.AudioFormat):
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
    if isinstance(format, bv.AudioFormat):
        format = format.name
    return np.dtype(format_dtypes[format])


def get_format(
    name: Union[str, type, np.dtype],
    is_planar: Optional[bool] = None,
    available_formats: Optional[Iterable[typing.AudioFormat]] = None,
) -> bv.AudioFormat:
    # name is dtype such as "float32", or float
    if isinstance(name, str) and name not in format_dtypes or isinstance(name, type):
        name = np.dtype(name)
    if isinstance(name, np.dtype):
        assert name in dtype_formats, f"Input dtype `{name}` not in {dtype_formats}."
        name = dtype_formats[name]
        if is_planar is not None:
            name = name + ("p" if is_planar else "")
        else:
            assert available_formats is not None
            available_formats = [
                format.name if isinstance(format, typing.AudioFormat) else format for format in available_formats
            ]
            if name not in available_formats:
                opposite_format = "packed" if name.endswith("p") else "planar"
                logger.warning(f"Input format `{name}` not in {available_formats}, try {opposite_format} format.")
                name = name.rstrip("p") if name.endswith("p") else name + "p"
            assert name in available_formats, f"Input format `{name}` not in {available_formats}."
    return AudioFormat[name].value


template = get_template("format")
for name, format in audio_formats.items():
    decodecs = get_codecs(name, "r")
    encodecs = get_codecs(name, "w")
    dtype = get_format_dtype(name)
    getattr(AudioFormat, name).__doc__ = template.render(
        format=format, decodecs=decodecs, encodecs=encodecs, dtype=dtype
    )
