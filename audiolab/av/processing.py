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


import av
import numpy as np
from numpy.typing import DTypeLike

from audiolab.av import filter
from audiolab.av.format import get_format
from audiolab.av.typing import FilterSpec


def aformat(
    dtype: str | type | np.dtype | None = None,
    is_planar: bool = False,
    sample_format: str | av.AudioFormat | None = None,
    sample_rate: int | None = None,
    to_mono: bool = False,
) -> FilterSpec:
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    kwargs = {}
    if dtype is not None:
        kwargs["sample_fmts"] = get_format(dtype, is_planar).name
    if sample_format is not None:
        kwargs["sample_fmts"] = sample_format.name if isinstance(sample_format, av.AudioFormat) else sample_format
    if sample_rate is not None:
        kwargs["sample_rates"] = sample_rate
    if to_mono:
        kwargs["channel_layouts"] = "mono"
    return filter.aformat(**kwargs)


def build_filter_chain(
    filters: list[FilterSpec] | None = None,
    *,
    dtype: DTypeLike | None = None,
    is_planar: bool = False,
    sample_format: str | av.AudioFormat | None = None,
    sample_rate: int | None = None,
    to_mono: bool = False,
    add_format: bool | None = None,
) -> list[FilterSpec] | None:
    chain = [] if filters is None else list(filters)
    if add_format is None:
        add_format = dtype is not None or sample_format is not None or sample_rate is not None or to_mono
    if add_format:
        chain.append(aformat(dtype, is_planar, sample_format, sample_rate, to_mono))
    return chain or None
