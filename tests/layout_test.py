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

import pytest

from audiolab.av.layout import AudioLayout, audio_layouts


class TestLayout:

    @pytest.mark.parametrize("name, layout", audio_layouts.items())
    def test_layout_name(self, name, layout):
        _layout = AudioLayout[name]
        assert _layout.value == layout
        assert _layout.nb_channels == len(layout.channels)
