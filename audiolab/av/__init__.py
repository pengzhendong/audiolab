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

from typing import Any, Optional, Union

import bv
import numpy as np

from audiolab.av import filter
from audiolab.av.codec import Decodec, Encodec, canonical_names, decodecs, encodecs
from audiolab.av.container import ContainerFormat, container_formats, extension_formats
from audiolab.av.format import AudioFormat, audio_formats, get_codecs, get_format, get_format_dtype
from audiolab.av.frame import clip, from_ndarray, split_audio_frame, to_ndarray
from audiolab.av.graph import AudioGraph
from audiolab.av.info import Info
from audiolab.av.layout import AudioLayout, audio_layouts, standard_channel_layouts
from audiolab.av.utils import load_url


def aformat(
    dtype: Optional[Union[str, type, np.dtype]] = None,
    is_planar: bool = False,
    format: Optional[Union[str, bv.AudioFormat]] = None,
    rate: Optional[int] = None,
    to_mono: bool = False,
):
    kwargs = {}
    if dtype is not None:
        kwargs["sample_fmts"] = get_format(dtype, is_planar).name
    if format is not None:
        if isinstance(format, bv.AudioFormat):
            format = format.name
        kwargs["sample_fmts"] = format
    if rate is not None:
        kwargs["sample_rates"] = rate
    if to_mono:
        kwargs["channel_layouts"] = "mono"
    return filter.aformat(**kwargs)


def info(file: Any, stream_id: int = 0) -> Info:
    return Info(file, stream_id)


__all__ = [
    "AudioFormat",
    "AudioGraph",
    "AudioLayout",
    "ContainerFormat",
    "Decodec",
    "Encodec",
    "Filter",
    "Info",
    "aformat",
    "audio_formats",
    "audio_layouts",
    "canonical_names",
    "clip",
    "container_formats",
    "decodecs",
    "encodecs",
    "extension_formats",
    "from_ndarray",
    "get_codecs",
    "get_format",
    "get_format_dtype",
    "info",
    "load_url",
    "split_audio_frame",
    "standard_channel_layouts",
    "to_ndarray",
]
