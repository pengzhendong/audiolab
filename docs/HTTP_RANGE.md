# Optional HTTP byte-range reading

Existing URL calls keep their original behavior. `cache_url=False` (the
existing default) passes the URL to the decoder; `cache_url=True` keeps the
existing whole-file download/cache path. Neither path is removed.

## Opt in

```python
from audiolab import load_audio

audio, rate = load_audio(
    "https://example.com/audio.wav",
    offset=120.25,
    duration=10.0,
    http_range=True,
    cache_url=False,
)
```

`Reader(..., http_range=True)` also supports streaming processed chunks.
Only HTTP(S) URLs use the new transport. Backend selection remains SoundFile,
then PyAV, or the explicit `backends` list. The audio backend and existing
processing still perform parsing, decoding, trimming, resampling and downmixing;
the transport does not assume a WAV header layout or audio sample rate.

Each reader owns a requests session and closes it when the reader closes,
including on initialization failure. Requests have finite byte ranges and a
10-second timeout. The first 64 KiB GET obtains the object length without HEAD.
Subsequent reads fetch blocks of at most 1 MiB. A per-source LRU holds at most
four blocks (at most 4 MiB of encoded data), and is discarded on close. This is
not the process-wide `AudioCache`. Enabling both `http_range` and `cache_url`
raises `ValueError` instead of silently choosing one.

## Server requirements and errors

The server must return 206, exact `Content-Range`/payload lengths, identity
encoding, and a strong ETag. Subsequent requests use `If-Match` and validate the
ETag and total length again. Redirects are supported; the original signed query
URL is passed unchanged. Signed URLs are not included in transport error text.

If the **initial** response ignores Range (200), or has no strong ETag, the
convenience API closes that response/session and falls back to the existing URL
path. It does not buffer the whole ignored response. This fallback does not
promise byte-range savings. Invalid ranges, short/oversized responses, errors or
object changes after admission fail closed; there is no mid-read URL fallback
that might mix versions. Reader also checks transport failure when a decoder
swallows a file callback exception, so eager loading cannot report a truncated
array as successful audio.

## Explicit file-like API

```python
from audiolab import HTTPRangeSource, info, load_audio

with HTTPRangeSource(
    "https://example.com/audio.wav",
    block_size=256 * 1024,
    cache_blocks=4,
    timeout=20,
) as source:
    metadata = info(source)
    metadata.close()
    source.seek(0)
    audio, rate = load_audio(source, offset=1.0, duration=2.0)
```

This API raises `RangeNotSupported` (an `OSError`, available from
`audiolab.reader.http_range`) on initial non-admission rather than falling back.
Caller-provided sources remain caller-owned. It implements `read`, `readinto`,
`tell`, and SET/CUR/END seeks, including seeking beyond EOF and back to zero.
`read()` without a size returns the remaining bytes as a normal binary file
would: the returned result may be large, while network requests and the cache
remain bounded. Use bounded reads/Reader for bounded decoded memory.

Use separate sources/readers for concurrent decoders. Individual source file
operations are serialized, but sharing a cursor among decoders is not a
multi-reader API. No shared mutable decoder or global URL cache is introduced.

## Validation and performance scope

The local HTTP tests cover finite ranges, connection reuse, bounded cache,
seek/read equivalence, redirect/signed-query handling, initial fallback,
ETag/length changes, ignored ranges after admission, short/oversized responses,
timeouts, cleanup, concurrent independent readers, and decoder failure handling.
Audio comparisons include PCM U8/16/24/32, FLOAT/DOUBLE WAV, WAVEX/RF64,
large JUNK headers, FLAC, MP3, MP4 and WebM, fractional offsets, EOF, stereo
conversion and 44.1/48 kHz to 16 kHz resampling. They compare arrays to local
reads using the same backend configuration, not a new numerical decoder.

The final implementation was also tested in a CPU-only environment with
three remote native 16 kHz WAV test files (ABBA, six fresh calls per method per case).
Timings include signing, complete `load_audio`, reader and session cleanup;
imports, corpus selection and PCM hashing are excluded equally. All 36 reads
matched previously saved frame counts and PCM SHA256 values exactly, with
independent saved-report/source-digest/median recomputation.

| Case | Default URL median | Opt-in range median | URL / range |
| --- | ---: | ---: | ---: |
| Selected audio clips | 1.580 s | 0.840 s | 1.88 |
| First 10 s | 0.773 s | 0.841 s | 0.92 |
| Last 10 s to EOF | 1.406 s | 0.756 s | 1.86 |

Both transport and effective backend selection differ on these WAV objects.
These are **not** isolated evidence for one optimization, a guaranteed speedup,
or application throughput. Remote server caches were uncontrolled; original
URL wire bytes were not measured. The results justify keeping this opt-in,
particularly since first reads did not improve. Benchmark your own deployment,
including cleanup, using the script below.

```bash
python benchmarks/http_range.py 'https://example.com/audio.wav' --offset 120 --duration 10
```

The benchmark reads the same audio using default URL and opt-in transport in
ABBA order, checks shape/rate/PCM equality, and reports end-to-end medians.
URLs and audio contents are not printed. If decoder selection differs and PCM
is not exact, it fails rather than calling that performance-only improvement.
