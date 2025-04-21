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
from bv.filter import filters_available

from audiolab import av
from audiolab.av import filter
from audiolab.av.format import format_dtypes, get_format


class TestFilter:

    @pytest.mark.parametrize("name", filters_available)
    def test_filter(self, name):
        _name, args, kwargs = getattr(filter, name)()
        assert _name == name
        assert args is None
        assert kwargs == {}

    def test_aformat(self):
        for is_planar in (True, False):
            for dtype in format_dtypes.values():
                format = get_format(dtype, is_planar)
                assert av.aformat(dtype=np.dtype(dtype), is_planar=is_planar)[2] == {"sample_fmts": format.name}
                assert av.aformat(dtype=np.dtype(dtype).name, is_planar=is_planar)[2] == {"sample_fmts": format.name}

        for rate in (8000, 16000, 24000, 48000):
            assert av.aformat(rate=rate)[2] == {"sample_rates": str(rate)}

        assert av.aformat(to_mono=False)[2] == {}
        assert av.aformat(to_mono=True)[2] == {"channel_layouts": "mono"}
