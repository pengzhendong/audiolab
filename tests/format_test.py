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

import numpy as np
import pytest

from audiolab.av.format import AudioFormat, audio_formats, format_dtypes, get_format, get_format_dtype


class TestFormat:

    @pytest.mark.parametrize("name, format", audio_formats.items())
    def test_format(self, name, format):
        _format = AudioFormat[name]
        assert _format.value == format

    @pytest.mark.parametrize("name, dtype", format_dtypes.items())
    def test_get_format(self, name, dtype):
        format = AudioFormat[name].value
        is_planar = name.endswith("p")
        assert get_format(name) == format
        if is_planar:
            assert get_format(dtype, available_formats=[format.packed]).name == format.packed.name
        else:
            assert get_format(dtype, available_formats=[format.planar]).name == format.planar.name
        assert get_format(np.dtype(dtype), is_planar) == format
        assert get_format(np.dtype(dtype).name, is_planar) == format

    @pytest.mark.parametrize("name, dtype", format_dtypes.items())
    def test_get_format_dtype(self, name, dtype):
        format = AudioFormat[name].value
        assert get_format_dtype(name) == np.dtype(dtype)
        assert get_format_dtype(format) == np.dtype(dtype)
