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

from threading import Lock
from typing import ClassVar


class AudioCache:
    """Thread-safe, insertion-ordered cache for encoded audio bytes."""

    max_bytes = 128 * 1024 * 1024
    max_entry_bytes = 16 * 1024 * 1024
    max_entries = 64

    _entries: ClassVar[dict[str, bytes]] = {}
    _size_bytes = 0
    _lock = Lock()

    @classmethod
    def get(cls, key: str) -> bytes | None:
        with cls._lock:
            value = cls._entries.pop(key, None)
            if value is not None:
                cls._entries[key] = value
            return value

    @classmethod
    def put(cls, key: str, value: bytes) -> None:
        if len(value) > min(cls.max_bytes, cls.max_entry_bytes):
            return

        with cls._lock:
            previous_value = cls._entries.pop(key, None)
            if previous_value is not None:
                cls._size_bytes -= len(previous_value)

            while cls._entries and (
                len(cls._entries) >= cls.max_entries or cls._size_bytes + len(value) > cls.max_bytes
            ):
                oldest_key = next(iter(cls._entries))
                cls._size_bytes -= len(cls._entries.pop(oldest_key))

            cls._entries[key] = value
            cls._size_bytes += len(value)

    @classmethod
    def size_bytes(cls) -> int:
        with cls._lock:
            return cls._size_bytes

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._entries.clear()
            cls._size_bytes = 0
