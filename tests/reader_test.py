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

from audiolab.av.filter import aresample, atempo
from audiolab.av.utils import generate_ndarray
from audiolab.reader import Reader, aformat, load_audio
from audiolab.writer import save_audio


class TestReader:
    @pytest.fixture
    def nb_channels(self):
        return 1

    @pytest.fixture
    def rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_reader(self, nb_channels, rate, duration):
        frame_size_ms = 50
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(
                nb_channels, int(rate * duration), np.int16, always_2d
            )
            save_audio(bytes_io, ndarray, rate=rate)

            reader = Reader(bytes_io, frame_size_ms=frame_size_ms, always_2d=always_2d)
            assert reader.channels == nb_channels
            assert reader.codec.name == "pcm_s16le"
            assert reader.duration == duration
            assert reader.precision == 16
            assert reader.rate == rate
            assert np.allclose(ndarray, reader.load_audio()[0])

    def test_load_audio(self, nb_channels, rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(
                nb_channels, int(rate * duration), np.int16, always_2d
            )
            save_audio(bytes_io, ndarray, rate=rate)

            audio, rate = load_audio(bytes_io, always_2d=always_2d)
            assert audio.dtype == np.int16
            if always_2d:
                assert audio.shape == (nb_channels, int(rate * duration))
            else:
                assert audio.ndim == 1
                assert audio.shape[0] == int(rate * duration)
            assert rate == rate
            assert np.allclose(ndarray, audio)

    def test_load_audio_with_filters(self, nb_channels, rate, duration):
        for ratio in (0.9, 1.1):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
            save_audio(bytes_io, ndarray, rate=rate)

            audio, rate = load_audio(bytes_io, filters=[atempo(ratio), aresample(8000)])
            assert audio.dtype == np.int16
            assert audio.shape[0] == nb_channels
            assert rate == 8000
            assert np.isclose(audio.shape[1] / rate, duration / ratio, atol=0.05)

        bytes_io = BytesIO()
        ndarray = generate_ndarray(2, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, rate=rate)

        audio, rate = load_audio(
            bytes_io, filters=[aformat(dtype=np.float32, rate=8000, to_mono=True)]
        )
        assert audio.dtype == np.float32
        assert audio.shape == (1, int(rate * duration))
        assert rate == 8000
