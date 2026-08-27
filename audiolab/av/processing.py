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

from typing import List, Optional, Union

import av
import numpy as np
from numpy.typing import DTypeLike

from audiolab.av import filter
from audiolab.av.format import get_format
from audiolab.av.typing import Filter


def aformat(
    dtype: Optional[Union[str, type, np.dtype]] = None,
    is_planar: bool = False,
    format: Optional[Union[str, av.AudioFormat]] = None,
    rate: Optional[int] = None,
    to_mono: bool = False,
) -> Filter:
    kwargs = {}
    if dtype is not None:
        kwargs["sample_fmts"] = get_format(dtype, is_planar).name
    if format is not None:
        kwargs["sample_fmts"] = format.name if isinstance(format, av.AudioFormat) else format
    if rate is not None:
        kwargs["sample_rates"] = rate
    if to_mono:
        kwargs["channel_layouts"] = "mono"
    return filter.aformat(**kwargs)


def build_filter_chain(
    filters: Optional[List[Filter]] = None,
    *,
    dtype: Optional[DTypeLike] = None,
    is_planar: bool = False,
    format: Optional[Union[str, av.AudioFormat]] = None,
    rate: Optional[int] = None,
    to_mono: bool = False,
    add_format: Optional[bool] = None,
) -> Optional[List[Filter]]:
    chain = [] if filters is None else list(filters)
    if add_format is None:
        add_format = dtype is not None or format is not None or rate is not None or to_mono
    if add_format:
        chain.append(aformat(dtype, is_planar, format, rate, to_mono))
    return chain or None
