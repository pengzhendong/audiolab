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

import errno
from fractions import Fraction
from typing import List, Optional

import av
import numpy as np
from av import filter

from audiolab.av import standard_channel_layouts
from audiolab.av.format import get_format
from audiolab.av.frame import from_ndarray, to_ndarray
from audiolab.av.typing import UINT32_MAX, AudioFormat, AudioFrame, AudioLayout, Dtype, Filter


class Graph(filter.Graph):
    def __init__(
        self,
        template: Optional[av.AudioStream] = None,
        rate: Optional[int] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        layout: Optional[AudioLayout] = None,
        channels: Optional[int] = None,
        time_base: Optional[Fraction] = None,
        filters: Optional[List[Filter]] = None,
        frame_size: Optional[int] = None,
        return_ndarray: bool = True,
    ):
        if template is not None:
            rate = template.sample_rate if rate is None else rate
            format = template.format if format is None else format
            layout = template.layout.name if layout is None else layout
            channels = template.channels if channels is None else channels
            time_base = template.time_base if time_base is None else time_base
        format = get_format(dtype, is_planar) if format is None else format
        format = format.name if isinstance(format, av.AudioFormat) else format
        time_base = Fraction(1, rate) if time_base is None else time_base
        if layout is None:
            layout = standard_channel_layouts[channels][0]
        abuffer = super().add_abuffer(None, rate, format, layout, channels, time_base=time_base)

        nodes = [abuffer]
        if filters is not None:
            for _filter in filters:
                name, args, kwargs = (
                    (_filter, None, {})
                    if isinstance(_filter, str)
                    else ((*_filter, {}) if len(_filter) == 2 else _filter)
                )
                nodes.append(super().add(name, args, **kwargs))
        nodes.append(super().add("abuffersink"))
        super().link_nodes(*nodes).configure()

        self.frame_size = None
        if frame_size is not None and frame_size > 0:
            self.frame_size = min(frame_size, UINT32_MAX)
            super().set_audio_frame_size(self.frame_size)

        self.rate = rate
        self.format = format
        self.layout = layout
        self.return_ndarray = return_ndarray

    def push(self, frame: AudioFrame):
        if isinstance(frame, tuple):
            frame, rate = frame
            assert rate == self.rate
        if isinstance(frame, np.ndarray):
            frame = from_ndarray(frame, self.format, self.layout, self.rate)
        super().push(frame)

    def pull(self, partial: bool = False, return_ndarray: Optional[bool] = None) -> AudioFrame:
        if partial:
            super().push(None)
        while True:
            try:
                frame = super().pull()
                if return_ndarray is None:
                    return_ndarray = self.return_ndarray
                yield (to_ndarray(frame), frame.rate) if return_ndarray else frame
            except av.EOFError:
                break
            except av.FFmpegError as e:
                if e.errno != errno.EAGAIN:
                    raise
                break
