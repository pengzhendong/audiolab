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

import numpy as np
import pytest

from audiolab.av.filter import atempo
from audiolab.av.utils import generate_ndarray
from audiolab.pipe import AudioPipe


class TestPipe:
    def test_pull_before_push_is_empty(self):
        pipe = AudioPipe(input_sample_rate=16_000)

        assert list(pipe.pull()) == []

    @pytest.fixture
    def nb_channels(self):
        return 1

    @pytest.fixture
    def rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_audio_pipe(self, nb_channels, rate, duration):
        num_chunks = 5
        num_samples = int(rate * duration * num_chunks)
        for ratio in (0.9, 1.1):
            for always_2d in (True, False):
                pipe = AudioPipe(input_sample_rate=rate, filters=[atempo(ratio)], always_2d=always_2d)
                frames = []
                for idx in range(num_chunks):
                    pipe.push(generate_ndarray(nb_channels, int(rate * duration), np.int16))
                    for frame, _ in pipe.pull(partial=idx == num_chunks - 1):
                        frames.append(frame)
                audio = np.concatenate(frames, axis=1 if always_2d else 0)
                assert np.isclose(audio.shape[1 if always_2d else 0] / rate * ratio, num_samples / rate, atol=0.05)

    def test_filters_are_only_added_when_requested(self, rate):
        pipe = AudioPipe(input_sample_rate=rate)
        assert pipe.filters is None

        filters = [atempo(1.1)]
        pipe = AudioPipe(input_sample_rate=rate, filters=filters, output_sample_rate=8000)

        assert len(filters) == 1
        assert len(pipe.filters) == 2

    def test_audio_pipe_applies_backpressure_until_output_is_drained(self, rate):
        pipe = AudioPipe(input_sample_rate=rate, frame_size=4, max_buffered_bytes=16)
        chunk = np.zeros((1, 8), dtype=np.int16)
        pipe.push(chunk)

        with pytest.raises(BufferError, match="pull"):
            pipe.push(np.zeros((1, 1), dtype=np.int16))

        list(pipe.pull())
        pipe.push(chunk)

    def test_audio_pipe_reset_releases_buffered_graph(self, rate):
        pipe = AudioPipe(input_sample_rate=rate)
        pipe.push(np.zeros((1, 8), dtype=np.int16))

        pipe.reset()

        assert pipe.graph is None
        assert pipe.buffered_bytes == 0
