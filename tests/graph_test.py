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

from audiolab.av import aformat
from audiolab.av.graph import Graph
from audiolab.av.utils import generate_ndarray


class TestGraph:
    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_push_pull(self, sample_rate):
        duration = 0.5
        frame_size = 1024
        filters = [aformat(dtype=np.int16, sample_rate=16000)]
        num_samples = int(sample_rate * duration)
        graph = Graph(
            sample_rate=sample_rate,
            dtype=np.float32,
            layout="mono",
            filters=filters,
            frame_size=frame_size,
        )
        assert graph.sample_rate == sample_rate
        ndarray = generate_ndarray(1, num_samples, np.float32)
        graph.push(ndarray)
        frames = []
        for frame, rate in graph.pull(partial=True):
            assert rate == 16000
            frames.append(frame)
        samples = np.concatenate(frames, axis=1)
        assert samples.shape[1] == 16000 * duration

    def test_constructor_validates_required_audio_parameters(self, sample_rate):
        with pytest.raises(ValueError, match="sample_rate is required"):
            Graph(dtype=np.float32, layout="mono")
        with pytest.raises(ValueError, match="dtype or sample_format is required"):
            Graph(sample_rate=sample_rate, layout="mono")
        with pytest.raises(ValueError, match="frame_size must be positive"):
            Graph(sample_rate=sample_rate, dtype=np.float32, layout="mono", frame_size=0)
