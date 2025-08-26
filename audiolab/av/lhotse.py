# Copyright (c) 2021 Piotr Żelasko
# From https://github.com/lhotse-speech/lhotse/blob/master/lhotse/caching.py
#      https://github.com/lhotse-speech/lhotse/blob/master/lhotse/utils.py
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

import logging
from io import BytesIO
from threading import Lock
from typing import Callable, Dict, Optional

from smart_open import open as sm_open


class AudioCache:
    """
    Cache of 'bytes' objects with audio data.
    It is used to cache the "command" type audio inputs.

    The cache size is limited to max 100 elements and 500MB of audio.

    A global dict `__cache_dict` (static member variable of class AudioCache)
    is holding the wavs as 'bytes' arrays.
    The key is the 'source' identifier (i.e. the command for loading the data).

    Thread-safety is ensured by a threading.Lock guard.
    """

    max_cache_memory: int = 500 * 1e6  # 500 MB
    max_cache_elements: int = 100  # 100 audio files

    __cache_dict: Dict[str, bytes] = {}
    __cache_memory: int = 0
    __lock: Lock = Lock()

    @classmethod
    def try_cache(cls, key: str) -> Optional[bytes]:
        """
        Test if 'key' is in the chache. If yes return the bytes array,
        otherwise return None.
        """

        with cls.__lock:
            if key in cls.__cache_dict:
                return cls.__cache_dict[key]
            else:
                return None

    @classmethod
    def add_to_cache(cls, key: str, value: bytes):
        """
        Add the new (key,value) pair to cache.
        Possibly free some elements before adding the new pair.
        The oldest elements are removed first.
        """

        if len(value) > cls.max_cache_memory:
            return

        with cls.__lock:
            # limit cache elements and memory
            while (
                len(cls.__cache_dict) >= cls.max_cache_elements
                or len(value) + cls.__cache_memory > cls.max_cache_memory
            ):
                # remove oldest elements from cache
                # (dict pairs are sorted according to insertion order)
                removed_key = next(iter(cls.__cache_dict))
                removed_value = cls.__cache_dict.pop(removed_key)
                cls.__cache_memory -= len(removed_value)

            # store the new (key,value) pair
            cls.__cache_dict[key] = value
            cls.__cache_memory += len(value)
            print(cls.__cache_memory)

    @classmethod
    @property
    def cache_memory(cls) -> int:
        """
        Return size of AudioCache values in bytes.
        """
        return cls.__cache_memory

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the cache, remove the data.
        """
        with cls.__lock:
            cls.__cache_dict.clear()
            cls.__cache_memory = 0


class SmartOpen:
    """Wrapper class around smart_open.open method

    The smart_open.open attributes are cached as classed attributes - they play the role of singleton pattern.

    The SmartOpen.setup method is intended for initial setup.
    It imports the `open` method from the optional `smart_open` Python package,
    and sets the parameters which are shared between all calls of the `smart_open.open` method.

    If you do not call the setup method it is called automatically in SmartOpen.open with the provided parameters.

    The example demonstrates that instantiating S3 `session.client` once,
    instead using the defaults and leaving the smart_open creating it every time
    has dramatic performance benefits.
    """

    transport_params: Optional[Dict] = None
    smart_open: Optional[Callable] = None

    @classmethod
    def setup(cls, transport_params: Optional[dict] = None):
        if cls.transport_params is not None and cls.transport_params != transport_params:
            logging.warning(
                f"SmartOpen.setup second call overwrites existing transport_params with new version"
                f"\t\n{cls.transport_params}\t\nvs\t\n{transport_params}"
            )
        cls.transport_params = transport_params
        cls.smart_open = sm_open

    @classmethod
    def open(cls, uri, mode="rb", transport_params=None, **kwargs):
        if cls.smart_open is None:
            cls.setup(transport_params=transport_params)
        transport_params = transport_params if transport_params else cls.transport_params
        return cls.smart_open(
            uri,
            mode=mode,
            transport_params=transport_params,
            **kwargs,
        )


def load_url(url: str, cache: bool = False) -> BytesIO:
    """
    Load an audio file from a URL.

    Args:
        url (str): The URL of the audio file.
        cache (bool): Whether to cache the audio file.
    Returns:
        The audio bytes.
    """
    audio_bytes = AudioCache.try_cache(url) if cache else None
    if audio_bytes is None:
        with SmartOpen.open(url, "rb") as f:
            audio_bytes = f.read()
        if cache:
            AudioCache.add_to_cache(url, audio_bytes)
    return BytesIO(audio_bytes)
