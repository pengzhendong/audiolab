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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from audiolab.reader.backend.backend import Backend
from audiolab.reader.backend.pyav import PyAV
from audiolab.reader.backend.soundfile import SoundFile
from audiolab.reader.backend.wave import Wave

BACKENDS: Dict[str, Type[Backend]] = {
    "av": PyAV,
    "pyav": PyAV,
    "sf": SoundFile,
    "soundfile": SoundFile,
    "wave": Wave,
}
DEFAULT_BACKENDS: Tuple[str, ...] = ("soundfile", "pyav")


def resolve_backends(names: Optional[Sequence[str]] = None) -> List[Type[Backend]]:
    names = DEFAULT_BACKENDS if names is None else names
    unknown = [name for name in names if name not in BACKENDS]
    if unknown:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown audio backend {unknown[0]!r}. Available backends: {available}")
    return [BACKENDS[name] for name in names]


def _get_position(source: Any):
    tell = getattr(source, "tell", None)
    if tell is None:
        return None
    try:
        return tell()
    except (OSError, ValueError):
        return None


def _restore_position(source: Any, position):
    if position is None:
        return
    seek = getattr(source, "seek", None)
    if seek is None:
        return
    try:
        seek(position)
    except (OSError, ValueError):
        pass


def open_backend(
    source: Any,
    frame_size: Optional[int] = None,
    forced_decoding: bool = False,
    names: Optional[Sequence[str]] = None,
) -> Backend:
    backend_types = resolve_backends(names)
    initial_position = _get_position(source)
    last_error = None

    for backend_type in backend_types:
        _restore_position(source, initial_position)
        backend = None
        keep_backend = False
        try:
            backend = backend_type(source, frame_size, forced_decoding)
            if backend.duration is not None or isinstance(backend, PyAV):
                keep_backend = True
                return backend
        except Exception as error:
            last_error = error
        finally:
            if backend is not None and not keep_backend:
                backend.close()

    _restore_position(source, initial_position)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No audio backend could read the source")
