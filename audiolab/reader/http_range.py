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

"""Opt-in, bounded HTTP byte-range transport for existing audio backends."""

import io
import operator
import re
from collections import OrderedDict
from threading import RLock
from urllib.parse import urlsplit


class RangeNotSupported(OSError):
    """The initial response cannot establish a validated range source."""


class HTTPRangeSource(io.RawIOBase):
    """Seekable HTTP(S) file with a private session and bounded read cache.

    Requires finite 206 responses and a stable strong ETag. No HEAD request or
    whole-file preload is performed. Each instance owns its session; use a
    separate instance for each concurrent decoder. ``read()`` can return the
    remaining file, like a normal binary file, but network requests and cached
    blocks remain bounded. Explicit users receive errors instead of fallback.
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int = 1024 * 1024,
        cache_blocks: int = 4,
        initial_size: int = 64 * 1024,
        timeout: float = 10,
    ):
        super().__init__()
        self._session = None
        self._failure = False
        self._blocks = OrderedDict()
        self._lock = RLock()
        for name, value in (("block_size", block_size), ("cache_blocks", cache_blocks), ("initial_size", initial_size)):
            if operator.index(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if urlsplit(url).scheme not in {"http", "https"}:
            raise ValueError("HTTPRangeSource requires an HTTP(S) URL")
        self._url = url
        self._block_size = operator.index(block_size)
        self._cache_blocks = operator.index(cache_blocks)
        self._initial_size = min(operator.index(initial_size), self._block_size)
        self._timeout = timeout
        self._position = 0
        self._length = None
        self._etag = None
        import requests

        self._session = requests.Session()
        try:
            self._fetch(0, self._initial_size)
        except BaseException:
            self.close()
            raise

    def _fetch(self, start: int, size: int) -> bytes:
        import requests

        end = start + size - 1
        if self._length is not None:
            end = min(end, self._length - 1)
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self._etag is not None:
            headers["If-Match"] = self._etag
        try:
            with self._session.get(
                self._url, headers=headers, stream=True, allow_redirects=True, timeout=self._timeout
            ) as response:
                if response.status_code == 200 and self._length is None:
                    raise RangeNotSupported("Server ignored the initial byte range")
                if response.status_code != 206:
                    raise OSError(f"HTTP range request returned status {response.status_code}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise OSError("HTTP range response must use identity encoding")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if match is None:
                    raise OSError("Invalid HTTP Content-Range")
                first, last, length = map(int, match.groups())
                if length <= 0 or first != start or last != min(end, length - 1) or last < first:
                    raise OSError("HTTP range response does not match the requested bytes")
                etag = response.headers.get("ETag", "")
                if len(etag) < 2 or not etag.startswith('"') or not etag.endswith('"'):
                    error = RangeNotSupported if self._length is None else OSError
                    raise error("HTTP range source requires a strong ETag")
                if self._length is not None and (length != self._length or etag != self._etag):
                    raise OSError("HTTP range source changed while reading")
                # Reading one extra byte detects oversized responses without buffering
                # an ignored Range's whole object. raw.read does not decompress bytes.
                data = response.raw.read(last - first + 2)
                if len(data) != last - first + 1:
                    raise OSError("HTTP range response has an incorrect payload length")
                self._length, self._etag = length, etag
        except (requests.RequestException, OSError) as error:
            self._failure = True
            self.close()
            if isinstance(error, RangeNotSupported):
                raise
            # Transport errors can contain signed query parameters. Do not expose
            # the underlying URL or exception chain in this public source's errors.
            raise OSError("HTTP range read failed") from None
        except Exception:
            self._failure = True
            self.close()
            raise OSError("HTTP range read failed") from None
        self._blocks[start] = data
        self._blocks.move_to_end(start)
        while len(self._blocks) > self._cache_blocks:
            self._blocks.popitem(last=False)
        return data

    def raise_if_failed(self) -> None:
        """Propagate transport failure even if a decoder swallowed an I/O callback error."""
        if self._failure:
            raise OSError("HTTP range read failed")

    def readable(self) -> bool:
        self._checkClosed()
        return True

    def seekable(self) -> bool:
        self._checkClosed()
        return True

    def tell(self) -> int:
        with self._lock:
            self._checkClosed()
            return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        with self._lock:
            self._checkClosed()
            offset = operator.index(offset)
            if whence == io.SEEK_SET:
                position = offset
            elif whence == io.SEEK_CUR:
                position = self._position + offset
            elif whence == io.SEEK_END:
                position = self._length + offset
            else:
                raise ValueError("Invalid seek origin")
            if position < 0:
                raise ValueError("Negative seek position")
            self._position = position
            return position

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            self._checkClosed()
            size = -1 if size is None else operator.index(size)
            remaining = max(0, self._length - self._position)
            size = remaining if size < 0 else min(size, remaining)
            chunks = []
            while size:
                block = next(
                    (
                        (start, data)
                        for start, data in reversed(self._blocks.items())
                        if start <= self._position < start + len(data)
                    ),
                    None,
                )
                if block is None:
                    start = self._position
                    block = start, self._fetch(start, self._block_size)
                start, data = block
                self._blocks.move_to_end(start)
                count = min(size, start + len(data) - self._position)
                chunks.append(data[self._position - start : self._position - start + count])
                self._position += count
                size -= count
            return b"".join(chunks)

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        if view.readonly:
            raise TypeError("readinto requires a writable buffer")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def close(self) -> None:
        with self._lock:
            try:
                if self._session is not None:
                    self._session.close()
                    self._session = None
                self._blocks.clear()
            finally:
                super().close()
