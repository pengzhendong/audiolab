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

from audiolab.av import aformat
from audiolab.av.filter import aresample, atempo
from audiolab.av.utils import generate_ndarray
from audiolab.reader import Reader, StreamReader, load_audio
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
            assert reader.channels == nb_channels
            assert reader.codec.name == "pcm_s16le"
            assert reader.duration == duration
            assert reader.precision == 16
            assert reader.rate == sample_rate
            assert np.allclose(ndarray, reader.load_audio()[0])

    def test_load_audio(self, nb_channels, sample_rate, duration):
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(sample_rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, sample_rate)

            audio, rate = load_audio(bytes_io, always_2d=always_2d)
            assert audio.dtype == np.int16
            if always_2d:
                assert audio.shape[0] == nb_channels
                assert audio.shape[1] == int(sample_rate * duration)
            else:
                assert audio.ndim == 1
                assert audio.shape[0] == int(sample_rate * duration)
            assert rate == sample_rate
            assert np.allclose(ndarray, audio)

    def test_load_audio_with_filters(self, nb_channels, sample_rate, duration):
        for ratio in [0.9, 1.1]:
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(sample_rate * duration), np.int16)
            save_audio(bytes_io, ndarray, sample_rate)

            audio, rate = load_audio(bytes_io, filters=[atempo(ratio), aresample(8000)])
            assert audio.dtype == np.int16
            assert audio.shape[0] == nb_channels
            assert rate == 8000
            assert np.isclose(audio.shape[1] / rate, duration / ratio, atol=0.05)

        bytes_io = BytesIO()
        ndarray = generate_ndarray(2, int(sample_rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate)

        audio, rate = load_audio(bytes_io, filters=[aformat(dtype=np.float32, rate=8000, to_mono=True)])
        assert audio.dtype == np.float32
        assert audio.shape[0] == 1
        assert audio.shape[1] == int(rate * duration)
        assert rate == 8000

    def test_stream_reader(self, nb_channels, sample_rate, duration):
        for always_2d in (True,):
            bytes_io = BytesIO()
            # ndarray = generate_ndarray(nb_channels, int(sample_rate * duration), np.int16, always_2d)
            ndarray, sample_rate = load_audio("/tmp/zfsv3/sata11/13121765827/data/code/audiolab/sample1.wav")
            save_audio(bytes_io, ndarray, sample_rate, container_format="webm")

            # opus 拼上之后需要重新从头解码，pcm 呢？
            frames = []
            chunk_size = 1024
            reader = StreamReader()
            while True:
                chunk = bytes_io.read(chunk_size)
                if chunk is not None and len(chunk) > 0:
                    reader.push(chunk)
                for frame, _ in reader.pull(partial=chunk is None or len(chunk) < chunk_size):
                    frames.append(frame)
                if chunk is None or len(chunk) < chunk_size:
                    break
            ndarray = np.concatenate(frames, axis=1 if always_2d else 0)
            print(ndarray.shape[1] / sample_rate)
            save_audio("/tmp/zfsv3/sata11/13121765827/data/code/audiolab/tmp.wav", ndarray, sample_rate)


# Input File     : 'sample1.wav'
# Channels       : 1
# Sample Rate    : 48000
# Precision      : 16-bit
# Duration       : 00:00:02.000 = 96000 samples ~ 150 CDDA sectors
# File Size      : 192.1 kB
# Bit Rate       : 768 kbps
# Sample Encoding: PCM signed 16-bit little-endian
# Comments       :
#     encoder: Lavf62.3.100
