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

        assert pipe._processor is None
        assert pipe.buffered_bytes == 0

    def test_audio_pipe_uses_high_level_speed_and_pitch_controls(self, rate):
        time = np.arange(rate, dtype=np.float32) / rate
        tone = (0.5 * np.sin(2 * np.pi * 440 * time)).reshape(1, -1)
        pipe = AudioPipe(
            input_sample_rate=rate,
            speed=1.25,
            pitch_shift=12,
            frame_size=256,
        )

        pipe.push(tone)
        chunks = list(pipe.pull(partial=True))
        audio = np.concatenate([chunk for chunk, _ in chunks], axis=1)

        frequencies = np.fft.rfftfreq(audio.shape[1], 1 / rate)
        dominant_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(audio[0] * np.hanning(audio.shape[1]))))]
        assert dominant_frequency == pytest.approx(880, abs=2)
        assert audio.shape[1] / rate == pytest.approx(0.8, abs=0.025)

    def test_soxr_streaming_is_independent_of_input_chunk_boundaries(self, rate):
        audio = generate_ndarray(2, rate, np.float32)

        whole = AudioPipe(input_sample_rate=rate, output_sample_rate=8000, frame_size=None)
        whole.push(audio)
        expected = np.concatenate([chunk for chunk, _ in whole.pull(partial=True)], axis=1)

        chunked = AudioPipe(input_sample_rate=rate, output_sample_rate=8000, frame_size=None)
        actual_chunks = []
        for offset in range(0, audio.shape[1], 317):
            chunked.push(audio[:, offset : offset + 317])
            actual_chunks.extend(chunk for chunk, _ in chunked.pull())
        actual_chunks.extend(chunk for chunk, _ in chunked.pull(partial=True))
        actual = np.concatenate(actual_chunks, axis=1)

        assert whole._processor.graph is None
        assert chunked._processor.graph is None
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected, atol=1e-6)
