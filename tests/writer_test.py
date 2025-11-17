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

from io import BytesIO

import numpy as np
import pytest

from audiolab.av.utils import generate_ndarray
from audiolab.reader import info
from audiolab.writer import Writer, save_audio


class TestWriter:
    @pytest.fixture
    def nb_channels(self):
        return 1

    @pytest.fixture
    def rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_writer(self, nb_channels, rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            # always int16 for pcm_s16le even if dtype of ndarray is float32
            ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16, always_2d)
            writer = Writer(bytes_io, rate, channels=1)
            writer.write(ndarray)
            writer.close()

            _info = info(bytes_io)
            assert _info.channels == nb_channels
            assert "signed 16" in _info.codec.lower()
            assert _info.duration == duration
            assert _info.precision == 16
            assert _info.rate == rate

    def test_save_audio(self, nb_channels, rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, rate, container_format="webm")

            _info = info(bytes_io)
            assert _info.channels == nb_channels
            assert _info.codec == "Opus"
            assert np.isclose(_info.duration, duration + 0.007, atol=0.001)  # Pre-skip / Encoder Delay for opus
            assert _info.precision == 32  # always float32 for opus
            assert _info.rate == 48000  # always 48k for opus
