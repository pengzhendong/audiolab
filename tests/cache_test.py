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

from audiolab.reader.cache import AudioCache


class TestAudioCache:
    def setup_method(self):
        AudioCache.clear()

    def teardown_method(self):
        AudioCache.clear()

    def test_replacing_an_entry_updates_memory_usage(self):
        AudioCache.put("source", b"123")
        AudioCache.put("source", b"45")

        assert AudioCache.get("source") == b"45"
        assert AudioCache.size_bytes() == 2

    def test_reading_an_entry_refreshes_its_eviction_order(self, monkeypatch):
        monkeypatch.setattr(AudioCache, "max_bytes", 4)
        AudioCache.put("first", b"12")
        AudioCache.put("second", b"34")

        assert AudioCache.get("first") == b"12"
        AudioCache.put("third", b"56")

        assert AudioCache.get("first") == b"12"
        assert AudioCache.get("second") is None
        assert AudioCache.get("third") == b"56"

    def test_oversized_entries_are_not_retained(self, monkeypatch):
        monkeypatch.setattr(AudioCache, "max_entry_bytes", 2)

        AudioCache.put("large", b"123")

        assert AudioCache.get("large") is None
        assert AudioCache.size_bytes() == 0
