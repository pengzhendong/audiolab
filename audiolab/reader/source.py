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

from io import BytesIO
from shutil import copyfileobj
from tempfile import SpooledTemporaryFile
from typing import Any
from urllib.parse import urlsplit

from audiolab.av.typing import Seconds
from audiolab.reader.cache import AudioCache

URL_REQUEST_TIMEOUT = 10
URL_SPOOL_MAX_BYTES = 8 * 1024 * 1024
URL_COPY_CHUNK_BYTES = 1024 * 1024


def _http_get(url: str):
    import requests

    return requests.get(url, allow_redirects=True, timeout=URL_REQUEST_TIMEOUT, stream=True)


def _open_remote(url: str):
    from smart_open import open as smart_open

    return smart_open(url, "rb")


def load_url(url: str, cache: bool = False):
    audio_bytes = AudioCache.get(url) if cache else None
    if audio_bytes is not None:
        return BytesIO(audio_bytes)

    destination = SpooledTemporaryFile(max_size=URL_SPOOL_MAX_BYTES, mode="w+b")  # noqa: SIM115
    try:
        if urlsplit(url).scheme in {"http", "https"}:
            with _http_get(url) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=URL_COPY_CHUNK_BYTES):
                    if chunk:
                        destination.write(chunk)
        else:
            with _open_remote(url) as source:
                copyfileobj(source, destination, length=URL_COPY_CHUNK_BYTES)
        destination.seek(0)
        if not cache:
            return destination

        audio_bytes = destination.read()
        AudioCache.put(url, audio_bytes)
        destination.close()
        return BytesIO(audio_bytes)
    except BaseException:
        destination.close()
        raise


def prepare_source(
    source: Any,
    *,
    offset: Seconds = 0.0,
    duration: Seconds | None = None,
    cache_url: bool = False,
):
    if isinstance(source, bytes):
        return BytesIO(source)
    if not isinstance(source, str) or "://" not in source:
        return source

    scheme = urlsplit(source).scheme
    if cache_url:
        return load_url(source, cache=cache_url)
    if scheme in {"http", "https"} or offset != 0 or duration is not None:
        return source
    if scheme:
        return load_url(source)
    return source
