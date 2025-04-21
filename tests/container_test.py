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

from audiolab.av.container import ContainerFormat, container_formats, extension_formats


class TestContainer:

    @pytest.mark.parametrize("name, format", container_formats.items())
    def test_input_container(self, name, format):
        _format = ContainerFormat[name]
        assert _format.value == format
        for extension in _format.extensions:
            assert extension in extension_formats
            assert name in extension_formats[extension]
