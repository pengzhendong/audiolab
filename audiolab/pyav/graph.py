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
from typing import List, Optional, Union

import av
import numpy as np
from av import AudioFormat, AudioFrame, AudioLayout, AudioStream
from av.filter import Graph

from audiolab.pyav.filters import Filter
from audiolab.pyav.utils import dtypes, from_ndarray, to_ndarray


class AudioGraph:
    def __init__(
        self,
        stream: Optional[AudioStream] = None,
        rate: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        format: Optional[Union[str, AudioFormat]] = None,
        layout: Optional[Union[str, AudioLayout]] = None,
        channels: Optional[int] = None,
        name: Optional[str] = None,
        time_base: Optional[Fraction] = None,
        filters: List[Filter] = [],
        frame_size: int = -1,
        return_ndarray: bool = True,
    ):
        if dtype is not None:
            format = dtypes[np.dtype(dtype)]
        self.filters = filters
        self.graph = Graph()
        if stream is None:
            abuffer = self.graph.add_abuffer(
                sample_rate=rate, format=format, layout=layout, channels=channels, name=name, time_base=time_base
            )
            self.rate = rate
            self.format = format.name if isinstance(format, AudioFormat) else format
            self.layout = layout
        else:
            abuffer = self.graph.add_abuffer(template=stream)
            self.rate = stream.rate
            self.format = stream.format.name
            self.layout = stream.layout
        nodes = [abuffer]
        for _filter in self.filters:
            name, args, kwargs = (
                (_filter, None, {}) if isinstance(_filter, str) else ((*_filter, {}) if len(_filter) == 2 else _filter)
            )
            nodes.append(self.graph.add(name, args, **kwargs))
        nodes.append(self.graph.add("abuffersink"))
        self.graph.link_nodes(*nodes).configure()

        if frame_size > 0:
            self.graph.set_audio_frame_size(frame_size)
        self.return_ndarray = return_ndarray

    def push(self, frame: Union[AudioFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            # [num_channels, num_samples]
            frame = from_ndarray(frame, self.format, self.layout, self.rate)
        self.graph.push(frame)

    def pull(self, partial: bool = False):
        if partial:
            self.graph.push(None)
        while True:
            try:
                frame = self.graph.pull()
                if self.return_ndarray:
                    # [num_channels, num_samples]
                    yield to_ndarray(frame), frame.rate
                else:
                    yield frame
            except av.EOFError:
                break
            except av.FFmpegError as e:
                if e.errno != errno.EAGAIN:
                    raise
                break
