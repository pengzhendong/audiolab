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
from audiolab.writer.backend.pyav import PyAV


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

    def test_save_audio(self, nb_channels, rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, rate, container_format="webm")

            _info = info(bytes_io)
            assert _info.num_channels == nb_channels
            assert _info.codec == "Opus"
            assert np.isclose(_info.duration, duration + 0.014, atol=0.001)  # Pre-skip / Encoder Delay for opus
            assert _info.bits_per_sample == 32  # always float32 for opus
            assert _info.sample_rate == 48000  # always 48k for opus

    def test_writer_validates_sample_rate_and_audio_shape(self, rate):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            Writer(BytesIO(), 0)

        writer = Writer(BytesIO(), rate)
        with pytest.raises(ValueError, match="audio must have shape"):
            writer.write(np.zeros((1, 2, 3), dtype=np.int16))
        writer.close()

    def test_writer_rejects_channel_count_changes(self, rate):
        writer = Writer(BytesIO(), rate)
        writer.write(np.zeros((2, 4), dtype=np.int16))

        with pytest.raises(ValueError, match="Expected 2 channels"):
            writer.write(np.zeros((1, 4), dtype=np.int16))
        writer.close()

    def test_explicit_close_propagates_backend_errors(self):
        class Backend:
            def close(self):
                raise RuntimeError("close failed")

        writer = object.__new__(Writer)
        writer.backend = Backend()

        with pytest.raises(RuntimeError, match="close failed"):
            writer.close()
        assert writer.backend is None

    def test_pyav_close_propagates_flush_errors_and_closes_container(self):
        class Stream:
            def encode(self):
                raise RuntimeError("flush failed")

        class Container:
            closed = False

            def close(self):
                self.closed = True

        container = Container()
        backend = object.__new__(PyAV)
        backend.destination = BytesIO()
        backend.container = container
        backend.stream = Stream()

        with pytest.raises(RuntimeError, match="flush failed"):
            backend.close()
        assert backend.container is None
        assert backend.stream is None
        assert backend.destination is None
        assert container.closed
