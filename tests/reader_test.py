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
from audiolab.reader import Reader, load_audio
from audiolab.writer import save_audio


class TestReader:

    @pytest.fixture
    def nb_channels(self):
        return 1

    @pytest.fixture
    def sample_rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_reader(self, nb_channels, sample_rate, duration):
        frame_size_ms = 50
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(sample_rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, sample_rate)

            reader = Reader(bytes_io, frame_size_ms=frame_size_ms, always_2d=always_2d)
            assert reader.num_frames == duration * 1000 / frame_size_ms
            assert reader.channels == nb_channels
            assert reader.codec.name == "pcm_s16le"
            assert reader.duration == duration
            assert reader.precision == 16
            assert reader.rate == sample_rate
            samples = np.concatenate(list(frame for frame, _ in reader), axis=1 if always_2d else 0)
            assert np.allclose(ndarray, samples)

    def test_load_audio(self, nb_channels, sample_rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(sample_rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, sample_rate)

            audio, rate = load_audio(bytes_io, always_2d=always_2d)
            if always_2d:
                assert audio.shape[0] == nb_channels
                assert audio.shape[1] == int(sample_rate * duration)
            else:
                assert audio.ndim == 1
                assert audio.shape[0] == int(sample_rate * duration)
            assert audio.dtype == np.int16
            assert rate == sample_rate
            assert np.allclose(ndarray, audio)

    def test_stream_reader(self):
        pass
