# Audio processing and filters

`audiolab` separates common transformations from advanced effects. Prefer high-level transform arguments whenever they express the result you need. They are easier to read, consistent across APIs, and automatically use the most efficient processing path available.

## Common transformations

The following arguments are supported by `load_audio`, `Reader`, and `StreamReader`. `AudioPipe` uses `output_sample_rate` instead of `sample_rate` because it has both an explicit input and output rate.

| Operation | Argument | Example | Output behavior |
| --- | --- | --- | --- |
| Resample | `sample_rate` | `sample_rate=16_000` | Returns the requested rate |
| Mix channels | `to_mono` | `to_mono=True` | Returns one channel |
| Convert sample dtype | `dtype` | `dtype=np.float32` | Returns the requested NumPy dtype |
| Change speed | `speed` | `speed=1.25` | About 20% shorter, with pitch preserved |
| Shift pitch | `pitch_shift` | `pitch_shift=-2` | Two semitones lower, with duration preserved |

These operations can be combined in one call:

```python
import numpy as np

from audiolab import load_audio

audio, sample_rate = load_audio(
    "input.mp3",
    sample_rate=16_000,
    to_mono=True,
    dtype=np.float32,
    speed=1.1,
    pitch_shift=2,
)
```

### Speed

`speed` is a positive playback-speed multiplier:

- `speed=1.0` leaves duration unchanged.
- `speed=2.0` produces approximately half the duration.
- `speed=0.5` produces approximately twice the duration.

Pitch is preserved. The exact number of output samples can differ by a small amount because time-domain processing and codec framing may introduce rounding or delay.

### Pitch

`pitch_shift` is measured in semitones:

- `pitch_shift=12` raises pitch by one octave.
- `pitch_shift=-12` lowers pitch by one octave.
- `pitch_shift=0` leaves pitch unchanged.

Duration and the reported output sample rate are preserved. Pitch and speed can be changed independently in the same operation.

### Array shape and dtype

Decoded audio is channels-first:

```text
(channels, samples)
```

Mono audio is still two-dimensional by default. Pass `always_2d=False` to `load_audio` or `Reader` if a one-dimensional mono array is more convenient.

Integer-to-float conversion is normalized to the floating-point audio range. Float-to-integer conversion clips values to the destination range rather than wrapping on overflow.

## Advanced filters

Use `filters` for effects that do not have a high-level argument, such as high-pass filtering, low-pass filtering, equalization, or dynamic processing. Filter helpers are generated from the audio filters available in the installed FFmpeg build.

```python
from audiolab import load_audio
from audiolab.av.filter import highpass, lowpass, volume

filters = [
    highpass(f=80),
    lowpass(f=8_000),
    volume(volume=0.8),
]

audio, sample_rate = load_audio("input.wav", filters=filters)
```

Filters run in list order. High-level output conversion can still be requested alongside them:

```python
import numpy as np

from audiolab import load_audio
from audiolab.av.filter import highpass

audio, sample_rate = load_audio(
    "input.wav",
    filters=[highpass(f=120)],
    sample_rate=16_000,
    to_mono=True,
    dtype=np.float32,
    speed=1.05,
)
```

The custom filters are applied first, followed by the requested speed, pitch, and final format conversion.

### Discover available filters

Filter availability depends on the FFmpeg libraries used by PyAV. You can inspect the current environment at runtime:

```python
from audiolab.av.filter import filters, highpass

print("highpass" in filters)
help(highpass)
```

For the complete option reference of a filter, use FFmpeg's own help command:

```bash
ffmpeg -h filter=highpass
```

### Filter argument forms

Generated helpers return filter specifications accepted by all processing APIs:

```python
from audiolab.av.filter import highpass

spec = highpass(f=200)
```

For specialized integrations, a specification may also be written as a filter name or as a tuple containing the filter name, positional argument string, and option mapping. Generated helpers are recommended because they are clearer and avoid hand-written string formatting.

## Performance guidance

- Express resampling, mono conversion, dtype conversion, speed, and pitch with their dedicated arguments.
- Reserve `filters` for effects that genuinely require a custom filter chain.
- Use `Reader` instead of `load_audio` when the complete decoded signal would consume too much memory.
- Use a stable chunk size with `AudioPipe`; very tiny chunks increase call overhead.
- Call `pull()` regularly after `push()` so processed chunks do not remain buffered.
- Call `pull(partial=True)` exactly once at end of input to flush delayed samples.

Unusable extreme speed and pitch combinations are rejected before decoding begins. This prevents enormous internal filter chains and backend-specific overflow errors.

For reproducible local comparisons of the optimized and advanced paths, run:

```bash
python benchmarks/processing.py
```

See the [streaming guide](streaming.md) for complete incremental examples.

[Back to README](../README.md)
