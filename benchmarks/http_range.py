"""Compare complete fresh URL / HTTP range reads without printing signed URLs."""

import argparse
import statistics
import time

import numpy as np

from audiolab import load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    timings = {False: [], True: []}
    reference = None
    for enabled in (False, True, True, False) * args.repeats:
        started = time.perf_counter()
        try:
            audio, rate = load_audio(
                args.url, offset=args.offset, duration=args.duration, dtype=np.float32, http_range=enabled
            )
        except Exception:
            raise RuntimeError("Audio read failed") from None
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = audio, rate
        else:
            expected, expected_rate = reference
            if rate != expected_rate or not np.array_equal(audio, expected):
                raise RuntimeError("Decoded PCM differs between transports")
        timings[enabled].append(elapsed)
    url, ranges = (statistics.median(timings[enabled]) for enabled in (False, True))
    print(f"URL median: {url:.6f}s; opt-in range median: {ranges:.6f}s; ratio: {url / ranges:.3f}")
    print(f"PCM exact; {len(timings[False])} fresh calls per transport (including cleanup)")


if __name__ == "__main__":
    main()
