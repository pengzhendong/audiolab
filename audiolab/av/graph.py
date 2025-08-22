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
from av.filter import Graph

from audiolab.av.format import get_format
from audiolab.av.frame import from_ndarray, to_ndarray
from audiolab.av.typing import AudioFormat, AudioFrame, AudioLayout, Dtype, Filter


class AudioGraph:
    def __init__(
        self,
        stream: Optional[av.AudioStream] = None,
        rate: Optional[int] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        layout: Optional[AudioLayout] = None,
        channels: Optional[int] = None,
        name: Optional[str] = None,
        time_base: Optional[Fraction] = None,
        filters: Optional[List[Filter]] = None,
        frame_size: int = -1,
        return_ndarray: bool = True,
    ):
        """
        Create an AudioGraph.

        Args:
            stream: The stream to create the AudioGraph from.
            rate: The sample rate of the input audio buffer.
            dtype: The data type of the input audio buffer.
            is_planar: Whether the input audio buffer is planar.
            format: The format of the input audio buffer.
            layout: The layout of the input audio buffer.
            channels: The number of channels of the input audio buffer.
            name: The name of the input audio buffer.
            time_base: The time base of the input audio buffer.
            filters: The filters to apply to the input audio buffer.
            frame_size: The frame size of the output audio buffer.
            return_ndarray: Whether to return the output audio frames as ndarrays.
        """
        self.filters = filters or []
        self.graph = Graph()
        if stream is None:
            if format is None:
                format = get_format(dtype, is_planar)
            abuffer = self.graph.add_abuffer(
                sample_rate=rate, format=format, layout=layout, channels=channels, name=name, time_base=time_base
            )
            self.rate = rate
            self.format = format.name if isinstance(format, av.AudioFormat) else format
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

    def push(self, frame: AudioFrame):
        """
        Push an audio frame to the graph.

        Args:
            frame: The audio frame to push.
                * shape of ndarray: [num_channels, num_samples]
        """
        if isinstance(frame, np.ndarray):
            frame = from_ndarray(frame, self.format, self.layout, self.rate)
        self.graph.push(frame)

    def pull(self, partial: bool = False, return_ndarray: Optional[bool] = None):
        """
        Pull an audio frame from the graph.

        Args:
            partial: Whether to pull a partial frame.
            return_ndarray: Whether to return the audio frame as an ndarray.
                * shape of ndarray: [num_channels, num_samples]
        """
        if partial:
            self.graph.push(None)
        while True:
            try:
                frame = self.graph.pull()
                if return_ndarray is None:
                    return_ndarray = self.return_ndarray
                if return_ndarray:
                    yield to_ndarray(frame), frame.rate
                else:
                    yield frame
            except av.EOFError:
                break
            except av.FFmpegError as e:
                if e.errno != errno.EAGAIN:
                    raise
                break
