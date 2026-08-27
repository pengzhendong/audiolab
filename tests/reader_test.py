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
from audiolab.reader import Reader, StreamReader, aformat, load_audio
from audiolab.reader.reader import _iter_audio_chunks
from audiolab.reader.source import URL_REQUEST_TIMEOUT
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
        frame_size = 1024
        for always_2d in (True, False):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16, always_2d)
            save_audio(bytes_io, ndarray, sample_rate=rate)

            reader = Reader(bytes_io, frame_size=frame_size, always_2d=always_2d)
            assert reader.num_channels == nb_channels
            assert "signed 16" in reader.codec.lower()
            assert reader.duration == duration
            assert reader.bits_per_sample == 16
            assert reader.sample_rate == rate

    def test_stream_reader_rejects_non_positive_frame_size(self):
        with pytest.raises(ValueError, match="frame_size must be positive"):
            StreamReader(frame_size=0)

    def test_reader_rejects_non_positive_frame_size(self):
        with pytest.raises(ValueError, match="frame_size must be positive"):
            Reader(BytesIO(), frame_size=0)

    def test_reader_requires_frame_size_when_padding(self):
        with pytest.raises(ValueError, match="frame_size is required"):
            Reader(BytesIO(), fill_value=0)

    def test_stream_reader_decodes_incremental_bytes_once(self, nb_channels, rate, duration):
        source = BytesIO()
        expected = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(source, expected, sample_rate=rate)
        encoded = source.getvalue()
        reader = StreamReader(frame_size=256)
        chunks = []

        for offset in range(0, len(encoded), 500):
            reader.push(encoded[offset : offset + 500])
            chunks.extend(reader.pull())
        chunks.extend(reader.pull(partial=True))

        decoded = np.concatenate([audio for audio, _ in chunks], axis=1)
        assert all(output_rate == rate for _, output_rate in chunks)
        assert np.array_equal(decoded, expected)
        with pytest.raises(RuntimeError, match="finalized"):
            reader.push(b"more")

    def test_reader_frame_size_without_filters(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate)

        with Reader(bytes_io, frame_size=1024) as reader:
            frames = list(reader)

        assert [frame.shape[1] for frame, _ in frames] == [1024] * 7 + [832]
        assert np.array_equal(np.concatenate([frame for frame, _ in frames], axis=1), ndarray)

    def test_load_audio_is_eager_even_when_frame_size_is_set(self, nb_channels, rate, duration):
        source = BytesIO()
        expected = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(source, expected, sample_rate=rate)

        audio, output_rate = load_audio(source, frame_size=1024)

        assert output_rate == rate
        assert np.array_equal(audio, expected)

    def test_load_audio(self, nb_channels, rate, duration):
        input_rate = rate
        for always_2d in (True, False):
            for offset in (0.0, 0.1, 0.2):
                for _duration in (None, 0.1, 0.2, 0.3):
                    bytes_io = BytesIO()
                    ndarray = generate_ndarray(nb_channels, int(input_rate * duration), np.int16, always_2d)
                    save_audio(bytes_io, ndarray, sample_rate=input_rate)

                    if _duration is None:
                        _duration = duration - offset
                    _duration = min(_duration, duration - offset)

                    audio, output_rate = load_audio(bytes_io, offset=offset, duration=_duration, always_2d=always_2d)
                    assert audio.dtype == np.int16
                    if always_2d:
                        assert audio.shape == (nb_channels, int(input_rate * _duration))
                        ndarray = ndarray[:, int(offset * input_rate) : int((offset + _duration) * input_rate)]
                    else:
                        assert audio.ndim == 1
                        assert audio.shape[0] == int(input_rate * _duration)
                        ndarray = ndarray[int(offset * input_rate) : int((offset + _duration) * input_rate)]
                    assert output_rate == input_rate
                    assert np.allclose(ndarray, audio)

    def test_load_audio_with_filters(self, nb_channels, rate, duration):
        input_rate = rate
        for ratio in (0.9, 1.1):
            bytes_io = BytesIO()
            ndarray = generate_ndarray(nb_channels, int(input_rate * duration), np.int16)
            save_audio(bytes_io, ndarray, sample_rate=input_rate)

            audio, output_rate = load_audio(bytes_io, filters=[atempo(ratio), aresample(8000)])
            assert audio.dtype == np.int16
            assert audio.shape[0] == nb_channels
            assert output_rate == 8000
            assert np.isclose(audio.shape[1] / output_rate, duration / ratio, atol=0.05)

        bytes_io = BytesIO()
        ndarray = generate_ndarray(2, int(input_rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=input_rate)

        audio, output_rate = load_audio(bytes_io, filters=[aformat(dtype=np.float32, sample_rate=8000, to_mono=True)])
        assert audio.dtype == np.float32
        assert audio.shape == (1, int(output_rate * duration))
        assert output_rate == 8000

    def test_dtype_conversion_with_custom_filters(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate)

        audio, output_rate = load_audio(bytes_io, filters=[atempo(1.0)], dtype=np.float32)

        assert audio.dtype == np.float32
        assert output_rate == rate

    def test_reader_owns_pyav_processing_graph(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate, container_format="webm")

        reader = Reader(
            bytes_io,
            filters=[aresample(8000)],
            frame_size=1024,
            backends=["pyav"],
        )
        frames = list(reader)

        assert reader.graph is not None
        assert not hasattr(reader.backend, "graph")
        assert frames
        assert all(output_rate == 8000 for _, output_rate in frames)
        reader.close()

    def test_pyav_offset_and_duration_are_preserved(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate, container_format="webm")

        audio, output_rate = load_audio(
            bytes_io,
            offset=0.1,
            duration=0.2,
            backends=["pyav"],
        )

        assert np.isclose(audio.shape[1] / output_rate, 0.2, atol=1 / output_rate)

    def test_filter_inputs_are_not_mutated(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate)
        filters = [atempo(1.1)]

        reader = Reader(bytes_io, filters=filters, sample_rate=8000)

        assert len(filters) == 1
        assert len(reader.filters) == 2
        reader.close()

        assert StreamReader().filters is None
        stream_reader = StreamReader(filters=filters, sample_rate=8000)
        assert len(filters) == 1
        assert len(stream_reader.filters) == 2

    def test_audio_chunking_uses_the_sample_axis(self):
        frame = np.arange(22, dtype=np.int16).reshape(2, 11)

        chunks = list(_iter_audio_chunks(frame, max_bytes=24))

        assert [chunk.shape[1] for chunk in chunks] == [4, 4, 3]
        assert np.array_equal(np.concatenate(chunks, axis=1), frame)

    def test_url_redirects_have_a_timeout(self, monkeypatch, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate)
        calls = []

        class Response:
            url = "https://cdn.example.com/audio.wav"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def raise_for_status(self):
                return None

            @property
            def content(self):
                return bytes_io.getvalue()

        def get(url, **kwargs):
            calls.append(("get", url, kwargs))
            return Response()

        source_module = __import__("audiolab.reader.source", fromlist=["source"])
        monkeypatch.setattr(source_module.requests, "get", get)

        reader = Reader("https://example.com/audio.wav")

        assert calls == [
            ("get", "https://example.com/audio.wav", {"allow_redirects": True, "timeout": URL_REQUEST_TIMEOUT}),
        ]
        reader.close()

    def test_non_http_urls_use_smart_open(self, monkeypatch):
        calls = []
        source_module = __import__("audiolab.reader.source", fromlist=["source"])

        def smart_open(url, mode):
            calls.append((url, mode))
            return BytesIO(b"encoded audio")

        def unexpected_request(*args, **kwargs):
            raise AssertionError("requests must only handle HTTP URLs")

        monkeypatch.setattr(source_module, "smart_open", smart_open)
        monkeypatch.setattr(source_module.requests, "get", unexpected_request)

        loaded = source_module.load_url("s3://bucket/audio.wav")

        assert loaded.read() == b"encoded audio"
        assert calls == [("s3://bucket/audio.wav", "rb")]
