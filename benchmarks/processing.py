"""Compare common high-level processing with the advanced filter path."""

import argparse
import statistics
import time

import numpy as np

from audiolab import AudioPipe
from audiolab.av import aformat


def process(audio: np.ndarray, sample_rate: int, *, optimized: bool) -> None:
    options = {
        "input_sample_rate": sample_rate,
        "frame_size": None,
    }
    if optimized:
        options.update(output_sample_rate=16_000, to_mono=True, dtype=np.int16)
    else:
        options["filters"] = [aformat(dtype=np.int16, sample_rate=16_000, to_mono=True)]

    pipe = AudioPipe(**options)
    pipe.push(audio)
    list(pipe.pull(partial=True))
    pipe.close()


def measure(audio: np.ndarray, sample_rate: int, *, optimized: bool, repeats: int) -> float:
    process(audio, sample_rate, optimized=optimized)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        process(audio, sample_rate, optimized=optimized)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.duration <= 0 or args.repeats <= 0:
        parser.error("duration and repeats must be positive")

    sample_rate = 48_000
    samples = round(sample_rate * args.duration)
    generator = np.random.default_rng(0)
    audio = generator.uniform(-0.5, 0.5, (2, samples)).astype(np.float32)

    optimized = measure(audio, sample_rate, optimized=True, repeats=args.repeats)
    advanced = measure(audio, sample_rate, optimized=False, repeats=args.repeats)
    print(f"optimized median: {optimized * 1000:.3f} ms")
    print(f"advanced median:  {advanced * 1000:.3f} ms")
    print(f"speedup:          {advanced / optimized:.2f}x")


if __name__ == "__main__":
    main()
