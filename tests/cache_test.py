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
