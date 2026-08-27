# Migration guide

This guide covers the breaking API cleanup after audiolab 0.5.x.

## Prefer high-level transformations

Code that manually assembled filters for speed, pitch, resampling, or mono conversion should use dedicated arguments:

```python
import numpy as np

from audiolab import load_audio

audio, sample_rate = load_audio(
    "input.wav",
    sample_rate=16_000,
    to_mono=True,
    dtype=np.float32,
    speed=1.25,
    pitch_shift=2,
)
```

The library selects the processing path automatically. There is no public engine, resampler, or quality selector.

## Import advanced helpers from audiolab.av

`Graph` and `aformat` are no longer top-level exports:

```python
from audiolab.av import Graph, aformat
```

Most applications do not need either helper. Custom effects remain available through `filters`:

```python
from audiolab import load_audio
from audiolab.av.filter import highpass

audio, sample_rate = load_audio("input.wav", filters=[highpass(f=120)])
```

## Remove low-level streaming format arguments

`AudioPipe` and `StreamReader` no longer accept `is_planar` or `sample_format`. NumPy output is always channels-first, and `dtype` is the high-level output sample type:

```python
import numpy as np

from audiolab import AudioPipe, StreamReader

pipe = AudioPipe(48_000, output_sample_rate=16_000, dtype=np.float32)
reader = StreamReader(sample_rate=16_000, dtype=np.float32)
```

## Treat partial pull as finalization

For `AudioPipe` and `StreamReader`, `pull(partial=True)` declares end-of-input and must be called exactly once. Use ordinary `pull()` while more input can still arrive.

```python
for chunk in input_chunks:
    pipe.push(chunk)
    consume_all(pipe.pull())

consume_all(pipe.pull(partial=True))
pipe.close()
```

## Account for layout-aware downmixing

Stereo-to-mono output keeps the normalized average behavior. Inputs with more than two channels now use their standard channel layout, including appropriate center and surround weights and exclusion of the LFE channel. This can change output levels compared with a plain arithmetic mean.

[Back to README](../README.md) · [Streaming guide](streaming.md) · [Processing and filters](filters.md)
