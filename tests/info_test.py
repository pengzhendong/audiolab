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

import pytest

from audiolab.reader.backend.registry import BACKENDS
from audiolab.reader.info import Info


class SeekableSource:
    def __init__(self, data: bytes):
        self.buffer = BytesIO(data)

    def read(self, size=-1):
        return self.buffer.read(size)

    def seek(self, offset, whence=0):
        return self.buffer.seek(offset, whence)

    def tell(self):
        return self.buffer.tell()


class WorkingBackend:
    positions = []

    def __init__(self, source, frame_size=None, forced_decoding=False):
        self.positions.append(source.tell())
        self.duration = 1.0
        self.closed = False

    def close(self):
        self.closed = True


class FailingBackend:
    def __init__(self, source, frame_size=None, forced_decoding=False):
        source.read(1)
        raise RuntimeError("backend failed")


class TestInfo:
    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown audio backend"):
            Info(BytesIO(b"invalid"), backends=["missing"])

    def test_seekable_source_is_reset_between_backend_attempts(self, monkeypatch):
        monkeypatch.setitem(BACKENDS, "failing", FailingBackend)
        monkeypatch.setitem(BACKENDS, "working", WorkingBackend)
        WorkingBackend.positions = []

        info = Info(SeekableSource(b"audio"), backends=["failing", "working"])

        assert WorkingBackend.positions == [0]
        info.close()
