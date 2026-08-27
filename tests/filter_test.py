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

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from av.filter import filters_available

from audiolab.av import aformat, filter
from audiolab.av.format import format_dtypes, get_format
from audiolab.av.utils import get_logger


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
                assert aformat(dtype=np.dtype(dtype), is_planar=is_planar)[2] == {"sample_fmts": format.name}
                assert aformat(dtype=np.dtype(dtype).name, is_planar=is_planar)[2] == {"sample_fmts": format.name}

        for rate in (8000, 16000, 24000, 48000):
            assert aformat(sample_rate=rate)[2] == {"sample_rates": str(rate)}

        assert aformat(to_mono=False)[2] == {}
        assert aformat(to_mono=True)[2] == {"channel_layouts": "mono"}

    def test_filter_without_option_introspection(self, monkeypatch):
        class Filter:
            def __init__(self, name):
                self.name = name
                self.description = "test filter"

        monkeypatch.setattr(
            filter,
            "filter",
            SimpleNamespace(filters_available={"test"}, Filter=Filter),
        )

        manager = filter.FilterManager()
        manager._generate_filter_data()

        assert manager._filter_data["test"] == {
            "name": "test",
            "description": "test filter",
            "options": [],
        }

    def test_missing_filter_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="not an FFmpeg audio filter"):
            _ = filter.definitely_missing

    def test_dynamic_loggers_do_not_accumulate_in_global_registry(self):
        registry = logging.Logger.manager.loggerDict
        before = set(registry)
        names = [f"audiolab.dynamic.{index}" for index in range(1000)]
        get_logger.cache_clear()
        try:
            loggers = [get_logger(name) for name in names]
            assert all(logger.name == name for logger, name in zip(loggers, names, strict=True))
            assert set(registry) == before
        finally:
            get_logger.cache_clear()
            for name in names:
                registry.pop(name, None)
