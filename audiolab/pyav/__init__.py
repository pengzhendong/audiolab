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

from typing import Any, Union

import numpy as np
from av import Codec, codecs_available
from av.codec.codec import UnknownCodecError

from audiolab.pyav import codecs, filters, formats
from audiolab.pyav.codecs import AudioCodec
from audiolab.pyav.filters import Filter
from audiolab.pyav.formats import AudioFormat
from audiolab.pyav.graph import AudioGraph
from audiolab.pyav.info import Info
from audiolab.pyav.utils import dtypes, from_ndarray, load_url, split_audio_frame, to_ndarray


def aformat(dtype: Union[str, np.dtype] = None, rate: int = None, to_mono: bool = False):
    kwargs = {}
    if dtype is not None:
        kwargs["sample_fmts"] = dtypes[np.dtype(dtype)]
    if rate is not None:
        kwargs["sample_rates"] = rate
    if to_mono:
        kwargs["channel_layouts"] = "mono"
    return filters.aformat(**kwargs)


def info(file: Any, stream_id: int = 0) -> Info:
    return Info(file, stream_id)


_codecs = {}
_formats = {}
for codec in codecs_available:
    try:
        codec = Codec(codec)
        if codec.type == "audio" and codec.audio_formats is not None:
            codec_name = codec.name
            if codec_name not in _codecs:
                _codecs[codec_name] = AudioCodec(codec_name)
                codecs.__dict__[codec_name] = _codecs[codec_name]
            for format in codec.audio_formats:
                format_name = format.name
                if format_name not in _formats:
                    _formats[format_name] = AudioFormat(format_name)
                    formats.__dict__[format_name] = _formats[format_name]
                codecs.__dict__[codec_name].formats.add(formats.__dict__[format_name])
                formats.__dict__[format_name].codecs.add(codecs.__dict__[codec_name])
            codecs.__dict__["codecs"] = _codecs
            formats.__dict__["formats"] = _formats
    except UnknownCodecError:
        pass


__all__ = [
    "AudioCodec",
    "AudioFormat",
    "AudioGraph",
    "Filter",
    "Info",
    "aformat",
    "info",
    "load_url",
    "split_audio_frame",
    "to_ndarray",
    "from_ndarray",
]
