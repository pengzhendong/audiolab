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
from audiolab.pipe import AudioPipe, get_template


class TestPipe:
    @pytest.fixture
    def nb_channels(self):
        return 1

    @pytest.fixture
    def rate(self):
        return 16000

    @pytest.fixture
    def duration(self):
        return 0.5

    def test_stream_template(self, nb_channels, rate, duration):
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        template = get_template(ndarray, rate)
        assert template.codec.name == "pcm_s16le"
        assert template.rate == rate
        assert template.channels == nb_channels

    def test_audio_pipe(self, nb_channels, rate, duration):
        num_frames = 5
        num_samples = int(rate * duration * num_frames)
        for ratio in (0.9, 1.1):
            pipe = AudioPipe(in_rate=rate, filters=[atempo(ratio)])
            frames = []
            for idx in range(num_frames):
                pipe.push(generate_ndarray(nb_channels, int(rate * duration), np.int16))
                for frame, _ in pipe.pull(partial=idx == num_frames - 1):
                    frames.append(frame)
            audio = np.concatenate(frames, axis=1)
            assert np.isclose(
                audio.shape[1] / rate * ratio, num_samples / rate, atol=0.05
            )
