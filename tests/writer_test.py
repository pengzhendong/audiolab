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
    def sample_rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_writer(self, sample_rate, duration):
        bytes_io = BytesIO()
        for always_2d in (True, False):
            ndarray = generate_ndarray(1, int(sample_rate * duration), np.int16, always_2d)
            writer = Writer(bytes_io, sample_rate, channels=1)
            writer.write(ndarray)
            writer.close()

            _info = info(bytes_io)
            assert _info.channels == 1
            assert _info.codec.name == "pcm_s16le"
            assert _info.duration == duration
            assert _info.precision == 16
            assert _info.rate == sample_rate

    def test_save_audio(self, sample_rate, duration):
        bytes_io = BytesIO()
        for always_2d in (True, False):
            ndarray = generate_ndarray(1, int(sample_rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, sample_rate)

            _info = info(bytes_io)
            assert _info.channels == 1
            assert _info.codec.name == "pcm_s16le"
            assert _info.duration == duration
            assert _info.precision == 16
            assert _info.rate == sample_rate
