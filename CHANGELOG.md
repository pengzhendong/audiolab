# Changelog

Notable user-facing changes are documented here. The project follows semantic versioning while the public API is evolving before version 1.0.

## Unreleased

### Added

- High-level `speed` and `pitch_shift` controls for eager, file-streaming, encoded-streaming, and PCM-streaming APIs.
- Automatic optimized processing for resampling, mono conversion, dtype conversion, speed, and pitch.
- Explicit buffering limits and deterministic cleanup for streaming objects.
- User guides for processing, advanced filters, and streaming lifecycles.
- PEP 561 `py.typed` marker for downstream type checkers.

### Changed

- `load_audio` now exposes an explicit keyword-only signature for editor completion and API discovery.
- Multichannel mono conversion uses layout-aware downmix coefficients; stereo conversion retains the lightweight normalized mix.
- `AudioPipe.buffered_bytes` includes PCM retained while waiting for a complete output frame.
- Extreme speed and pitch combinations fail early with `ValueError` instead of leaking backend exceptions.

### Removed

- Top-level `Graph` and `aformat` exports. Advanced low-level utilities remain under `audiolab.av`.
- Low-level `is_planar` and `sample_format` arguments from `AudioPipe` and `StreamReader`.

See the [migration guide](docs/migration.md) for examples.
