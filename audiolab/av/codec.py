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

from collections import defaultdict
from typing import Dict, Set

import bv
import numpy as np
from bv import codecs_available
from bv.codec.codec import UnknownCodecError

from audiolab.av.format import format_dtypes
from audiolab.av.typing import CodecEnum
from audiolab.av.utils import get_template

canonical_names: Dict[str, Set[str]] = defaultdict(set)
decodecs: Dict[str, bv.Codec] = {}
encodecs: Dict[str, bv.Codec] = {}
for codec in codecs_available:
    try:
        decoder_codec = bv.Codec(codec)
        if decoder_codec.type != "audio":
            continue
        if decoder_codec.audio_formats is not None:
            canonical_name = decoder_codec.canonical_name
            codec_name = decoder_codec.name
            canonical_names[canonical_name].add(codec_name)
            if codec_name not in decodecs:
                decodecs[codec_name] = decoder_codec

        encoder_codec = bv.Codec(codec, "w")
        if encoder_codec.audio_formats is not None:
            canonical_name = encoder_codec.canonical_name
            codec_name = encoder_codec.name
            canonical_names[canonical_name].add(codec_name)
            if codec_name not in encodecs:
                encodecs[codec_name] = encoder_codec
    except UnknownCodecError:
        pass
Decodec = CodecEnum("Decodec", decodecs)
Encodec = CodecEnum("Encodec", encodecs)


template = get_template("codec")
for name, codec in decodecs.items():
    getattr(Decodec, name).__doc__ = template.render(codec=codec, format_dtypes=format_dtypes, np=np)
for name, codec in encodecs.items():
    getattr(Encodec, name).__doc__ = template.render(codec=codec, format_dtypes=format_dtypes, np=np)
