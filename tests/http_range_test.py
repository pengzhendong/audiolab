"""Real local HTTP + decoder tests; no external credentials or services required."""

import io
import random
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlsplit

import numpy as np
import pytest
import requests
import soundfile as sf

from audiolab import HTTPRangeSource, Reader, load_audio
from audiolab.reader.http_range import RangeNotSupported
from audiolab.reader.source import prepare_source
from audiolab.writer.backend.pyav import PyAV as PyAVWriter


@pytest.fixture(scope="module")
def server():
    objects = {}
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            with suppress(BrokenPipeError, ConnectionResetError):
                super().handle()

        def log_message(self, *args):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            item = objects[path]
            item["count"] = item.get("count", 0) + 1
            calls.append((path, dict(self.headers), self.client_address))
            if item.get("redirect"):
                self.send_response(302)
                self.send_header("Location", item["redirect"])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = item["data"]
            mode = item.get("mode", "normal")
            if item["count"] > 1:
                mode = item.get("later_mode", mode)
            etag = item.get("etag", '"v1"')
            if mode == "change":
                etag = '"v2"'
            if mode == "precondition":
                self.send_response(412)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if match is None or mode == "ignore":
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)
                return
            start, end = map(int, match.groups())
            end = min(end, len(data) - 1)
            payload = data[start : end + 1]
            total = len(data) + (1 if mode == "resize" else 0)
            content_range = f"bytes {start}-{end}/{total}"
            if mode == "wrong_start":
                content_range = f"bytes {start + 1}-{end}/{total}"
            if mode == "malformed":
                content_range = "bytes */*"
            if mode == "short":
                payload = payload[:-1]
            if mode == "long":
                payload += b"x"
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload)))
            if mode != "no_range":
                self.send_header("Content-Range", content_range)
            if mode != "no_etag":
                self.send_header("ETag", 'W/"weak"' if mode == "weak" else etag)
            if mode == "encoding":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    counter = 0

    def add(data, **kwargs):
        nonlocal counter
        counter += 1
        path = f"/audio-{counter}"
        objects[path] = {"data": data, **kwargs}
        return f"http://127.0.0.1:{httpd.server_port}{path}?signature=keep-this-query"

    yield add, calls, objects
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@contextmanager
def source_for(server, data, **kwargs):
    add, _, _ = server
    with HTTPRangeSource(add(data), block_size=257, initial_size=31, cache_blocks=2, **kwargs) as source:
        yield source


def test_random_seek_read_matches_bytesio_and_bounds_cache(server):
    data = bytes(range(256)) * 100
    rng = random.Random(42)
    with source_for(server, data) as source:
        expected = io.BytesIO(data)
        for _ in range(200):
            target = rng.randrange(len(data) + 100)
            count = rng.randrange(800)
            assert source.seek(target) == expected.seek(target)
            assert source.read(count) == expected.read(count)
            assert source.tell() == expected.tell()
            assert len(source._blocks) <= 2
            assert sum(map(len, source._blocks.values())) <= 514
        source.seek(-300, io.SEEK_END)
        assert source.read() == data[-300:]
        source.seek(0)
        buffer = bytearray(500)
        assert source.readinto(buffer) == 500
        assert buffer == data[:500]
        assert source.seek(-2, io.SEEK_CUR) == 498
        assert source.read(0) == b""
        with pytest.raises(TypeError):
            source.readinto(b"readonly")
        assert source.tell() == 498
        with pytest.raises(ValueError):
            source.seek(-1)
        with pytest.raises(ValueError):
            source.seek(0, 99)


def test_finite_requests_cache_session_and_close(server):
    add, calls, _ = server
    url = add(b"a" * 5000)
    with HTTPRangeSource(url, block_size=100, initial_size=20) as source:
        source.read(10)
        before = len(calls)
        source.seek(0)
        source.read(10)
        assert len(calls) == before
        source.seek(200)
        source.read(250)
        requests_for_source = [c for c in calls if c[0] == urlsplit(url).path]
        assert all(re.fullmatch(r"bytes=\d+-\d+", c[1]["Range"]) for c in requests_for_source)
        assert all(c[1]["Accept-Encoding"] == "identity" for c in requests_for_source)
        assert all(c[1]["If-Match"] == '"v1"' for c in requests_for_source[1:])
        assert len({c[2] for c in requests_for_source}) == 1
    assert source.closed and source._session is None and not source._blocks
    source.close()
    for action in (source.tell, source.read, source.readable, source.seekable):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("mode", ["malformed", "wrong_start", "no_range", "short", "long", "encoding"])
def test_bad_initial_responses_fail_closed(server, mode):
    add, _, _ = server
    with pytest.raises(OSError):
        HTTPRangeSource(add(b"x" * 300, mode=mode), block_size=50)


@pytest.mark.parametrize("mode", ["change", "resize", "ignore", "precondition", "no_etag", "weak", "short"])
def test_midstream_failure_cannot_return_cached_or_mixed_bytes(server, mode):
    add, _, _ = server
    source = HTTPRangeSource(add(b"x" * 500, later_mode=mode), block_size=50, initial_size=20)
    source.seek(100)
    with pytest.raises(OSError):
        source.read(1)
    assert source.closed
    with pytest.raises(OSError):
        source.raise_if_failed()
    with pytest.raises(ValueError):
        source.read(1)


@pytest.mark.parametrize("mode", ["ignore", "no_etag", "weak"])
def test_initial_non_admission_falls_back_only_in_opt_in_api(server, mode):
    add, _, _ = server
    url = add(b"x" * 100, mode=mode)
    with pytest.raises(RangeNotSupported):
        HTTPRangeSource(url)
    assert prepare_source(url, http_range=True) == url


def test_defaults_and_conflicting_options(server):
    add, _, _ = server
    url = add(b"audio")
    assert prepare_source(url) is url
    with pytest.raises(ValueError, match="cannot"):
        prepare_source(url, cache_url=True, http_range=True)
    assert prepare_source(b"audio", http_range=True).read() == b"audio"
    assert prepare_source("/tmp/local.wav", http_range=True) == "/tmp/local.wav"


@pytest.mark.parametrize("kwargs", [{"block_size": 0}, {"cache_blocks": 0}, {"initial_size": -1}, {"timeout": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        HTTPRangeSource("https://example.invalid/audio", **kwargs)


def test_timeout_closes_session_and_hides_signed_url(monkeypatch):
    sessions = []

    class Session:
        closed = False

        def __init__(self):
            sessions.append(self)

        def get(self, *args, **kwargs):
            raise requests.Timeout("https://example.invalid/?secret=credential")

        def close(self):
            self.closed = True

    monkeypatch.setattr(requests, "Session", Session)
    with pytest.raises(OSError) as caught:
        HTTPRangeSource("https://example.invalid/?secret=credential")
    assert "credential" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert sessions[0].closed


def test_redirect_preserves_validated_range(server):
    add, calls, _ = server
    target = add(b"abc" * 500)
    redirect = add(b"", redirect=target)
    with HTTPRangeSource(redirect) as source:
        assert source.read(9) == b"abc" * 3
    target_calls = [c for c in calls if c[0] == urlsplit(target).path]
    assert target_calls[-1][1]["Range"] == "bytes=0-65535"


@pytest.mark.parametrize(
    "container,subtype",
    [
        ("WAV", "PCM_U8"),
        ("WAV", "PCM_16"),
        ("WAV", "PCM_24"),
        ("WAV", "PCM_32"),
        ("WAV", "FLOAT"),
        ("WAV", "DOUBLE"),
        ("WAVEX", "PCM_24"),
        ("RF64", "PCM_16"),
        ("FLAC", "PCM_24"),
    ],
)
@pytest.mark.parametrize("offset,duration", [(0, None), (0.03125, 0.1375), (0.5, None)])
def test_soundfile_pcm_offsets_eof_downmix_resampling(server, container, subtype, offset, duration):
    add, _, _ = server
    rng = np.random.default_rng(42)
    signal = rng.uniform(-0.8, 0.8, (44117, 2))
    encoded = io.BytesIO()
    sf.write(encoded, signal, 44100, format=container, subtype=subtype)
    data = encoded.getvalue()
    options = {
        "offset": offset,
        "duration": duration,
        "dtype": np.float32,
        "sample_rate": 16000,
        "to_mono": True,
        "always_2d": False,
        "read_size": 731,
    }
    expected, rate = load_audio(data, **options)
    actual, actual_rate = load_audio(add(data), http_range=True, **options)
    assert rate == actual_rate == 16000
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("container", ["mp3", "mp4", "webm"])
@pytest.mark.parametrize("offset,duration", [(0, None), (0.1, 0.2), (0.5, None)])
def test_pyav_compressed_backend_and_eof(server, container, offset, duration):
    add, _, _ = server
    encoded = io.BytesIO()
    t = np.arange(48017) / 48000
    signal = (20000 * np.sin(2 * np.pi * 437 * t)).astype(np.int16).reshape(1, -1)
    with PyAVWriter(encoded, 48000, container_format=container) as writer:
        writer.write(signal)
    data = encoded.getvalue()
    kwargs = {
        "offset": offset,
        "duration": duration,
        "dtype": np.float32,
        "to_mono": True,
        "sample_rate": 16000,
        "backends": ["pyav"],
    }
    expected, rate = load_audio(data, **kwargs)
    actual, actual_rate = load_audio(add(data), http_range=True, **kwargs)
    assert rate == actual_rate
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_decoder_cannot_publish_truncated_success_after_range_failure(server):
    add, _, _ = server
    encoded = io.BytesIO()
    sf.write(encoded, np.zeros(200000), 16000, format="WAV", subtype="FLOAT")
    with pytest.raises((OSError, ValueError)):
        load_audio(add(encoded.getvalue(), later_mode="change"), http_range=True)


def test_independent_concurrent_decoders_and_owned_source_lifecycle(server):
    add, _, _ = server
    encoded = io.BytesIO()
    sf.write(encoded, np.linspace(-0.5, 0.5, 320001), 16000, format="WAV", subtype="FLOAT")
    data = encoded.getvalue()
    url = add(data)

    def decode(i):
        expected, _ = load_audio(data, offset=i / 10, duration=0.2)
        with Reader(url, offset=i / 10, duration=0.2, http_range=True) as reader:
            owned = reader._owned_source
            actual = np.concatenate([chunk for chunk, _ in reader], axis=1)
        assert owned.closed
        np.testing.assert_array_equal(actual, expected)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(decode, range(16)))


def test_invalid_audio_releases_owned_source(server, monkeypatch):
    add, _, _ = server
    closed = []
    original = HTTPRangeSource.close

    def close(self):
        closed.append(self)
        original(self)

    monkeypatch.setattr(HTTPRangeSource, "close", close)
    with pytest.raises((OSError, ValueError)):
        Reader(add(b"not audio"), http_range=True)
    assert closed and all(source.closed for source in closed)


@pytest.mark.parametrize("mode", ["ignore", "no_etag", "weak"])
def test_fallback_decodes_original_url(server, mode):
    add, _, _ = server
    encoded = io.BytesIO()
    sf.write(encoded, np.linspace(-0.3, 0.3, 48000), 16000, format="WAV", subtype="PCM_16")
    data = encoded.getvalue()
    url = add(data, mode=mode)
    expected, rate = load_audio(url, duration=0.1)
    actual, actual_rate = load_audio(url, duration=0.1, http_range=True)
    assert rate == actual_rate
    np.testing.assert_array_equal(actual, expected)


def test_repeated_decoder_seeks_including_zero_and_nonstandard_wav_header(server):
    add, _, _ = server
    encoded = io.BytesIO()
    sf.write(encoded, np.linspace(-0.7, 0.7, 64000), 16000, format="WAV", subtype="PCM_24")
    original = encoded.getvalue()
    # A large legal JUNK chunk ensures no fixed 44-byte WAV layout assumptions.
    junk = b"JUNK" + (70000).to_bytes(4, "little") + b"\0" * 70000
    data = original[:12] + junk + original[12:]
    data = data[:4] + (len(data) - 8).to_bytes(4, "little") + data[8:]
    url = add(data)
    with (
        HTTPRangeSource(url, block_size=65536, cache_blocks=2) as source,
        sf.SoundFile(source) as decoder,
        sf.SoundFile(io.BytesIO(data)) as reference,
    ):
        for offset in [10000, 0, 63999, 0, 12345]:
            decoder.seek(offset)
            reference.seek(offset)
            np.testing.assert_array_equal(decoder.read(1000), reference.read(1000))
    assert source.closed
