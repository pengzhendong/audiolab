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

from audiolab.av import AudioGraph, aformat
from audiolab.av.utils import generate_ndarray


class TestGraph:
    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_push_pull(self, sample_rate):
        duration = 0.5
        filters = [aformat(dtype=np.int16, rate=16000)]
        num_samples = int(sample_rate * duration)
        for always_2d in (True, False):
            graph = AudioGraph(
                rate=sample_rate, dtype=np.float32, layout="mono", filters=filters
            )
            ndarray = generate_ndarray(1, num_samples, np.float32, always_2d)
            graph.push(ndarray)
            frames = []
            for frame, rate in graph.pull(True, True, always_2d):
                assert rate == 16000
                frames.append(frame)
            samples = np.concatenate(frames, axis=1 if always_2d else 0)
            assert samples.shape[1 if always_2d else 0] == 16000 * duration
