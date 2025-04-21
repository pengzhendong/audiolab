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

import pytest

from audiolab.av.codec import Decodec, Encodec, decodecs, encodecs
from audiolab.av.format import get_codecs


class TestCodec:

    @pytest.mark.parametrize("name, codec", decodecs.items())
    def test_decoder_codec(self, name, codec):
        _codec = Decodec[name]
        assert _codec.value == codec
        assert _codec.is_decoder
        assert _codec.mode == "r"
        assert _codec.type == "audio"
        assert _codec.audio_formats is not None
        for format in _codec.audio_formats:
            assert _codec.name in get_codecs(format.name, "r")

    @pytest.mark.parametrize("name, codec", encodecs.items())
    def test_encoder_codec(self, name, codec):
        _codec = Encodec[name]
        assert _codec.value == codec
        assert _codec.is_encoder
        assert _codec.mode == "w"
        assert _codec.type == "audio"
        assert _codec.audio_formats is not None
        for format in _codec.audio_formats:
            assert _codec.name in get_codecs(format.name, "w")
