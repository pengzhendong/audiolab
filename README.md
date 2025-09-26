# audiolab

[![PyPI](https://img.shields.io/pypi/v/audiolab)](https://pypi.org/project/audiolab/)
[![License](https://img.shields.io/github/license/pengzhendong/audiolab)](LICENSE)

A Python library for audio processing built on top of [PyAV](https://github.com/PyAV-Org/PyAV) (bindings for FFmpeg). audiolab provides a simple and efficient interface for loading, processing, and saving audio files.

## Features

- Load audio files in various formats (WAV, MP3, FLAC, AAC, etc.)
- Save audio files in different container formats
- Support for audio streaming and real-time processing
- Command-line interface for audio file inspection
- Support for audio transformations and filtering

## Installation

```bash
pip install audiolab
```

## Quick Start

### Load an audio file

```python
from audiolab import load_audio

# Load an audio file and convert to mono with specific sample rate
audio, rate = load_audio("audio.wav", rate=16000, to_mono=True)
print(f"Sample rate: {rate} Hz")
print(f"Audio shape: {audio.shape}")
```

### Save an audio file

```python
import numpy as np
from audiolab import save_audio

# Create a simple sine wave
rate = 44100
duration = 5
t = np.linspace(0, duration, rate * duration)
audio = np.sin(2 * np.pi * 440 * t)

# Save as WAV file
save_audio("tone.wav", audio, rate)
```

### Get audio file information

```python
from audiolab import info

# Get information about an audio file
print(info("audio.wav"))
```

### Command-line usage

```bash
# Get information about an audio file
audi audio.wav

# Show only specific information
audi -r -c audio.wav  # Show sample rate and channels only
audi -d audio.wav     # Show duration in hours, minutes and seconds
audi -D audio.wav     # Show duration in seconds
```

#### CLI Options

- `-S, --stream-id INTEGER`        The index of the stream to load (default: 0)
- `-f, --force-decoding`           Force decoding the audio file to get the duration
- `-t, --show-file-type`           Show detected file-type
- `-r, --show-sample-rate`         Show sample-rate
- `-c, --show-channels`            Show number of channels
- `-s, --show-samples`             Show number of samples (N/A if unavailable)
- `-d, --show-duration-hms`        Show duration in hours, minutes and seconds (N/A if unavailable)
- `-D, --show-duration-seconds`    Show duration in seconds (N/A if unavailable)
- `-b, --show-bits-per-sample`     Show number of bits per sample (N/A if not applicable)
- `-B, --show-bitrate`             Show the bitrate averaged over the whole file (N/A if unavailable)
- `-p, --show-precision`           Show estimated sample precision in bits
- `-e, --show-encoding`            Show the name of the audio encoding
- `-a, --show-comments`            Show file comments (annotations) if available
- `--help`                         Show this message and exit

If no specific options are selected, all information will be displayed by default.

## API Overview

### Core Functions

- `load_audio()`: Load audio from file or stream
- `save_audio()`: Save audio to file or stream
- `info()`: Get information about an audio file
- `encode()`: Transform audio to PCM bytestring

### Classes

- `Reader`: Read audio files with advanced options
- `StreamReader`: Read audio streams
- `Writer`: Write audio files with custom parameters

## Advanced Usage

### Apply filters during loading

```python
from audiolab import info, load_audio
from audiolab.av.filter import aresample, asetrate, atempo

# Speed perturbation
filters = [atempo(1.5)]
audio, rate = audiolab.load_audio("audio.wav", filters=filters)

# Pitch perturbation
ratio = 1.5
rate = info("audio.wav").rate
filters = [asetrate(rate * ratio), atempo(1 / ratio), aresample(rate)]
audio, rate = load_audio("audio.wav", filters=filters)
```

### Stream processing

```python
from audiolab import Reader

# Process audio in chunks
reader = Reader("audio.wav", frame_size=1024)
for chunk, rate in reader:
    print(chunk.shape)
```

## License

[Apache License 2.0](LICENSE)
