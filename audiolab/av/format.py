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

from collections.abc import Iterator
from functools import cache
from typing import Literal

import av
import numpy as np
from av import Codec, codecs_available
from av.codec.codec import UnknownCodecError
from numpy.typing import DTypeLike

from audiolab.av import typing

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
dtype_formats = {np.dtype(dtype): name for name, dtype in format_dtypes.items() if not name.endswith("p")}
audio_formats: dict[str, av.AudioFormat] = {name: av.AudioFormat(name) for name in format_dtypes}
AudioFormat = typing.AudioFormatEnum("AudioFormat", audio_formats)


@cache
def _codec_formats(mode: Literal["r", "w"]) -> dict[str, set[str]]:
    supported_codecs = {name: set() for name in audio_formats}
    for codec_name in codecs_available:
        try:
            codec = Codec(codec_name, mode)
            formats = codec.audio_formats
            if codec.type != "audio" or formats is None:
                continue
            for supported_format in formats:
                if supported_format.name in supported_codecs:
                    supported_codecs[supported_format.name].add(codec.name)
        except UnknownCodecError:
            pass
    return supported_codecs


def get_codecs(sample_format: typing.AudioFormatLike, mode: Literal["r", "w"] = "r") -> set[str]:
    if isinstance(sample_format, av.AudioFormat):
        sample_format = sample_format.name
    try:
        return set(_codec_formats(mode)[sample_format])
    except KeyError:
        raise ValueError(f"Unsupported audio sample format: {sample_format!r}") from None


def get_dtype(sample_format: typing.AudioFormatLike) -> np.dtype:
    if isinstance(sample_format, av.AudioFormat):
        sample_format = sample_format.name
    return _get_dtype(sample_format)


@cache
def _get_dtype(sample_format: str) -> np.dtype:
    try:
        return np.dtype(format_dtypes[sample_format])
    except KeyError:
        raise ValueError(f"Unsupported audio sample format: {sample_format!r}") from None


def get_format(
    dtype: DTypeLike,
    is_planar: bool | None = None,
    available_formats: Iterator[typing.AudioFormatLike] | None = None,
) -> av.AudioFormat:
    if (isinstance(dtype, str) and dtype not in format_dtypes) or isinstance(dtype, type):
        dtype = np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        try:
            dtype = dtype_formats[dtype]
        except KeyError:
            raise ValueError(f"Unsupported audio dtype: {dtype}") from None
        if is_planar is not None:
            dtype = dtype + ("p" if is_planar else "")
        else:
            if available_formats is None:
                raise ValueError("available_formats is required when is_planar is not specified")
            available_format_names = [
                sample_format.name if isinstance(sample_format, av.AudioFormat) else sample_format
                for sample_format in available_formats
            ]
            if dtype not in available_format_names:
                dtype = dtype.rstrip("p") if dtype.endswith("p") else dtype + "p"
    try:
        return audio_formats[dtype]
    except KeyError:
        raise ValueError(f"Unsupported audio sample format: {dtype!r}") from None
