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

from inspect import signature
from io import BytesIO

import av
import numpy as np
import pytest

from audiolab.av import aformat
from audiolab.av.filter import aresample, atempo
from audiolab.av.utils import generate_ndarray
from audiolab.reader import Reader, StreamReader, load_audio
from audiolab.reader.reader import DEFAULT_READ_FRAMES, _iter_audio_chunks
from audiolab.reader.source import URL_COPY_CHUNK_BYTES, URL_REQUEST_TIMEOUT
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

    def test_stream_reader_keeps_one_decoder_and_releases_consumed_input(
        self, monkeypatch, nb_channels, rate, duration
    ):
        source = BytesIO()
        expected = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(source, expected, sample_rate=rate)
        encoded = source.getvalue()
        stream_reader_module = __import__("audiolab.reader.stream_reader", fromlist=["stream_reader"])
        original_open = stream_reader_module.av.open
        open_calls = 0

        def counted_open(*args, **kwargs):
            nonlocal open_calls
            open_calls += 1
            return original_open(*args, **kwargs)

        monkeypatch.setattr(stream_reader_module.av, "open", counted_open)
        reader = StreamReader(frame_size=256)
        chunks = []
        for offset in range(0, len(encoded), 500):
            reader.push(encoded[offset : offset + 500])
            chunks.extend(reader.pull())
            assert reader.buffered_bytes <= 500
        chunks.extend(reader.pull(partial=True))

        decoded = np.concatenate([audio for audio, _ in chunks], axis=1)
        assert np.array_equal(decoded, expected)
        assert open_calls == 1
        assert reader.buffered_bytes == 0

    def test_stream_reader_falls_back_to_disk_for_seekable_containers(self, rate):
        source = BytesIO()
        expected = generate_ndarray(1, rate, np.int16)
        save_audio(source, expected, sample_rate=rate, container_format="mp4")
        encoded = source.getvalue()
        reader = StreamReader(frame_size=256)
        chunks = []

        for offset in range(0, len(encoded), 500):
            reader.push(encoded[offset : offset + 500])
            chunks.extend(reader.pull())
        chunks.extend(reader.pull(partial=True))

        decoded = np.concatenate([audio for audio, _ in chunks], axis=1)
        assert np.array_equal(decoded, expected)
        assert all(output_rate == rate for _, output_rate in chunks)
        assert reader.buffered_bytes == 0

    def test_stream_reader_can_drain_after_a_partially_consumed_pull(self, nb_channels, rate, duration):
        source = BytesIO()
        expected = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(source, expected, sample_rate=rate)
        reader = StreamReader(frame_size=256)
        reader.push(source.getvalue())

        final_pull = reader.pull(partial=True)
        first = next(final_pull)
        remaining = list(reader.pull())
        decoded = np.concatenate([first[0], *(audio for audio, _ in remaining)], axis=1)

        assert np.array_equal(decoded, expected)

    def test_stream_reader_rejects_input_after_decoder_failure(self, monkeypatch):
        stream_reader_module = __import__("audiolab.reader.stream_reader", fromlist=["stream_reader"])
        monkeypatch.setattr(
            stream_reader_module.av,
            "open",
            lambda *args, **kwargs: (_ for _ in ()).throw(av.InvalidDataError(22, "bad stream")),
        )
        reader = StreamReader()
        reader.push(b"broken")

        with pytest.raises(av.InvalidDataError):
            list(reader.pull())

        assert reader.buffered_bytes == 0
        assert reader._state.error is None
        with pytest.raises(RuntimeError, match="decoder has failed"):
            reader.push(b"more")

    def test_stream_reader_close_releases_error_and_graph(self):
        reader = StreamReader()
        reader._state.error = RuntimeError("retained traceback")
        reader._state.graph = object()

        reader.close()

        assert reader._state.error is None
        assert reader._state.graph is None
        assert reader._thread is None

    def test_stream_reader_supports_high_level_speed_and_pitch(self, rate):
        source = BytesIO()
        time = np.arange(rate, dtype=np.float32) / rate
        tone = (16_000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16).reshape(1, -1)
        save_audio(source, tone, sample_rate=rate)
        reader = StreamReader(dtype=np.float32, speed=1.25, pitch_shift=12, frame_size=256)
        reader.push(source.getvalue())

        chunks = list(reader.pull(partial=True))
        audio = np.concatenate([chunk for chunk, _ in chunks], axis=1)

        frequencies = np.fft.rfftfreq(audio.shape[1], 1 / rate)
        dominant_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(audio[0] * np.hanning(audio.shape[1]))))]
        assert dominant_frequency == pytest.approx(880, abs=2)
        assert audio.shape[1] / rate == pytest.approx(0.8, abs=0.025)

    def test_reader_bounds_default_read_size(self, rate):
        source = BytesIO()
        expected = generate_ndarray(1, DEFAULT_READ_FRAMES * 2 + 1, np.int16)
        save_audio(source, expected, sample_rate=rate)

        with Reader(source) as reader:
            first, output_rate = next(iter(reader))

        assert output_rate == rate
        assert first.shape == (1, DEFAULT_READ_FRAMES)

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

    def test_load_audio_does_not_copy_a_single_decoded_chunk(self, monkeypatch):
        expected = np.arange(12, dtype=np.int16).reshape(1, -1)

        class FakeReader:
            dtype = expected.dtype
            sample_rate = 16_000

            def __init__(self, source, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def __iter__(self):
                yield expected, self.sample_rate

        reader_module = __import__("audiolab.reader", fromlist=["reader"])
        monkeypatch.setattr(reader_module, "Reader", FakeReader)

        actual, output_rate = reader_module.load_audio("source")

        assert output_rate == FakeReader.sample_rate
        assert actual is expected

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

    def test_common_transforms_bypass_filter_graph(self, rate):
        source = BytesIO()
        left = np.linspace(-20_000, 20_000, rate, dtype=np.int16)
        right = np.linspace(10_000, -10_000, rate, dtype=np.int16)
        save_audio(source, np.stack((left, right)), sample_rate=rate)

        reader = Reader(
            source,
            dtype=np.float32,
            sample_rate=8000,
            to_mono=True,
            frame_size=256,
            backends=["wave"],
        )
        chunks = list(reader)

        assert reader._processor.graph is None
        assert chunks
        assert all(chunk.shape == (1, 256) for chunk, _ in chunks[:-1])
        assert all(output_rate == 8000 for _, output_rate in chunks)
        assert all(chunk.dtype == np.float32 for chunk, _ in chunks)
        reader.close()

    def test_fast_mono_conversion_averages_normalized_channels(self, rate):
        source = BytesIO()
        stereo = np.array([[16_384, -16_384], [0, 16_384]], dtype=np.int16)
        save_audio(source, stereo, sample_rate=rate)

        audio, _ = load_audio(source, dtype=np.float32, to_mono=True, backends=["wave"])

        expected = np.array([[0.25, 0.0]], dtype=np.float32)
        assert np.allclose(audio, expected, atol=1 / 32_768)

    @pytest.mark.parametrize(
        ("kwargs", "expected_frequency", "expected_duration"),
        [
            ({"speed": 1.25}, 440.0, 0.8),
            ({"speed": 4.0}, 440.0, 0.25),
            ({"pitch_shift": 12.0}, 880.0, 1.0),
            ({"pitch_shift": -12.0}, 220.0, 1.0),
            ({"speed": 1.25, "pitch_shift": 12.0}, 880.0, 0.8),
            ({"filters": [atempo(1.0)], "speed": 1.25, "pitch_shift": 12.0}, 880.0, 0.8),
        ],
    )
    def test_high_level_speed_and_pitch_controls(self, rate, kwargs, expected_frequency, expected_duration):
        source = BytesIO()
        time = np.arange(rate, dtype=np.float32) / rate
        tone = (16_000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16).reshape(1, -1)
        save_audio(source, tone, sample_rate=rate)

        audio, output_rate = load_audio(source, dtype=np.float32, backends=["wave"], **kwargs)

        windowed = audio[0] * np.hanning(audio.shape[1])
        frequencies = np.fft.rfftfreq(audio.shape[1], 1 / output_rate)
        dominant_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(windowed)))]
        assert output_rate == rate
        assert dominant_frequency == pytest.approx(expected_frequency, abs=2.0)
        assert audio.shape[1] / output_rate == pytest.approx(expected_duration, abs=0.025)

    @pytest.mark.parametrize(("name", "value"), [("speed", 0), ("pitch_shift", float("nan"))])
    def test_high_level_transforms_validate_finite_values(self, name, value):
        with pytest.raises(ValueError, match=name):
            Reader(BytesIO(), **{name: value})

    @pytest.mark.parametrize(("name", "value"), [("speed", 1e100), ("pitch_shift", 20_000)])
    def test_high_level_transforms_reject_unusable_extremes(self, name, value):
        with pytest.raises(ValueError, match=name):
            StreamReader(**{name: value})

    def test_processing_engines_are_not_exposed_in_public_reader_api(self):
        for reader_type in (Reader, StreamReader):
            parameters = signature(reader_type).parameters
            assert "engine" not in parameters
            assert "resampler" not in parameters
            assert "quality" not in parameters
        stream_parameters = signature(StreamReader).parameters
        assert "is_planar" not in stream_parameters
        assert "sample_format" not in stream_parameters

    def test_load_audio_has_an_explicit_keyword_api(self):
        parameters = signature(load_audio).parameters

        assert list(parameters) == [
            "source",
            "offset",
            "duration",
            "filters",
            "dtype",
            "sample_rate",
            "to_mono",
            "speed",
            "pitch_shift",
            "frame_size",
            "read_size",
            "cache_url",
            "always_2d",
            "fill_value",
            "backends",
        ]
        assert all(parameter.kind is parameter.KEYWORD_ONLY for parameter in list(parameters.values())[1:])

    def test_dtype_conversion_with_custom_filters(self, nb_channels, rate, duration):
        bytes_io = BytesIO()
        ndarray = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(bytes_io, ndarray, sample_rate=rate)

        audio, output_rate = load_audio(bytes_io, filters=[atempo(1.0)], dtype=np.float32)

        assert audio.dtype == np.float32
        assert output_rate == rate

    def test_reader_owns_and_releases_pyav_processing_graph(self, nb_channels, rate, duration):
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

        assert reader._processor._custom_graph
        assert reader._processor.graph is None
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

    def test_cached_url_redirects_have_a_timeout(self, monkeypatch, nb_channels, rate, duration):
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

            def iter_content(self, chunk_size):
                assert chunk_size == URL_COPY_CHUNK_BYTES
                yield bytes_io.getvalue()

        def get(url, **kwargs):
            calls.append(("get", url, kwargs))
            return Response()

        source_module = __import__("audiolab.reader.source", fromlist=["source"])
        monkeypatch.setattr(
            source_module,
            "_http_get",
            lambda url: get(
                url,
                allow_redirects=True,
                timeout=URL_REQUEST_TIMEOUT,
                stream=True,
            ),
        )

        reader = Reader("https://example.com/audio.wav", cache_url=True)

        assert calls == [
            (
                "get",
                "https://example.com/audio.wav",
                {"allow_redirects": True, "timeout": URL_REQUEST_TIMEOUT, "stream": True},
            ),
        ]
        reader.close()

    def test_http_urls_stream_without_preloading(self, monkeypatch):
        source_module = __import__("audiolab.reader.source", fromlist=["source"])
        monkeypatch.setattr(
            source_module,
            "load_url",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HTTP source was preloaded")),
        )

        url = "https://example.com/audio.wav"

        assert source_module.prepare_source(url) == url

    def test_non_http_urls_use_smart_open(self, monkeypatch):
        calls = []
        source_module = __import__("audiolab.reader.source", fromlist=["source"])

        def remote_open(url):
            calls.append(url)
            return BytesIO(b"encoded audio")

        monkeypatch.setattr(source_module, "_open_remote", remote_open)

        loaded = source_module.load_url("s3://bucket/audio.wav")

        assert loaded.read() == b"encoded audio"
        assert calls == ["s3://bucket/audio.wav"]
        loaded.close()

    def test_reader_closes_downloaded_source(self, monkeypatch, nb_channels, rate, duration):
        source = BytesIO()
        expected = generate_ndarray(nb_channels, int(rate * duration), np.int16)
        save_audio(source, expected, sample_rate=rate)
        downloaded = BytesIO(source.getvalue())
        source_module = __import__("audiolab.reader.source", fromlist=["source"])
        monkeypatch.setattr(source_module, "load_url", lambda url, cache=False: downloaded)

        reader = Reader("https://example.com/audio.wav", cache_url=True)
        reader.close()

        assert downloaded.closed

    def test_large_url_payload_spills_to_disk(self, monkeypatch):
        source_module = __import__("audiolab.reader.source", fromlist=["source"])

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"12345678"

        monkeypatch.setattr(source_module, "URL_SPOOL_MAX_BYTES", 4)
        monkeypatch.setattr(source_module, "_http_get", lambda url: Response())

        loaded = source_module.load_url("https://example.com/large.wav")

        assert loaded._rolled
        assert loaded.read() == b"12345678"
        loaded.close()
