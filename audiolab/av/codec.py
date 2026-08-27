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

import av
import numpy as np
from av import codecs_available
from av.codec.codec import UnknownCodecError

from audiolab.av.format import format_dtypes
from audiolab.av.typing import CodecEnum
from audiolab.av.utils import get_template

"""
$ ffmpeg -codecs
$ ffmpeg -decoders
$ ffmpeg -encoders
"""


class CodecCatalog:
    def __init__(self):
        self.aliases: dict[str, set[str]] = defaultdict(set)
        self.decoders: dict[str, av.Codec] = {}
        self.encoders: dict[str, av.Codec] = {}

        for codec in codecs_available:
            try:
                decoder_codec = av.Codec(codec)
                if decoder_codec.type != "audio":
                    continue
                if decoder_codec.audio_formats is not None:
                    canonical_name = decoder_codec.canonical_name
                    codec_name = decoder_codec.name
                    self.aliases[canonical_name].add(codec_name)
                    self.decoders.setdefault(codec_name, decoder_codec)

                encoder_codec = av.Codec(codec, "w")
                if encoder_codec.audio_formats is not None:
                    canonical_name = encoder_codec.canonical_name
                    codec_name = encoder_codec.name
                    self.aliases[canonical_name].add(codec_name)
                    self.encoders.setdefault(codec_name, encoder_codec)
            except UnknownCodecError:
                pass

        self.AudioDecoder = CodecEnum("AudioDecoder", self.decoders)
        self.AudioEncoder = CodecEnum("AudioEncoder", self.encoders)

        template = get_template("codec")
        for name, codec in self.decoders.items():
            getattr(self.AudioDecoder, name).__doc__ = template.render(codec=codec, format_dtypes=format_dtypes, np=np)
        for name, codec in self.encoders.items():
            getattr(self.AudioEncoder, name).__doc__ = template.render(codec=codec, format_dtypes=format_dtypes, np=np)


_catalog = CodecCatalog()
codec_aliases = _catalog.aliases
audio_decoders = _catalog.decoders
audio_encoders = _catalog.encoders
AudioDecoder = _catalog.AudioDecoder
AudioEncoder = _catalog.AudioEncoder
