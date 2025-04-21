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

import math
from typing import Any, List, Optional

from lhotse import Seconds

from audiolab.av import AudioGraph, Info, aformat, load_url, split_audio_frame
from audiolab.av.typing import AudioFormat, Dtype, Filter


class Reader(Info):
    def __init__(
        self,
        file: Any,
        stream_id: int = 0,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        filters: List[Filter] = [],
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = None,
        frame_size_ms: Optional[int] = None,
        return_ndarray: bool = True,
    ):
        # Open and seek url by ffmpeg may cause some issues.
        if isinstance(file, str) and "://" in file:
            file = load_url(file)

        super().__init__(file, stream_id)
        self.start_time = int(offset / self.stream.time_base)
        self.end_time = offset + duration if duration is not None else Seconds("inf")
        if self.start_time > 0:
            self.container.seek(self.start_time, any_frame=True, stream=self.stream)

        if not all([dtype is None, format is None, rate is None, to_mono is None]):
            filters.append(aformat(dtype, is_planar, format, rate, to_mono))
        if frame_size is None and frame_size_ms is not None:
            frame_size = frame_size_ms * self.stream.rate / 1000
        self.frame_size = frame_size
        self.graph = AudioGraph(
            stream=self.stream, filters=filters, frame_size=frame_size, return_ndarray=return_ndarray
        )

    @property
    def num_frames(self) -> int:
        # original number of frames(without aformat, aresample, etc.)
        return math.ceil(self.duration * self.rate / self.frame_size)

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            assert frame.time == float(frame.pts * self.stream.time_base)
            if frame.time > self.end_time:
                break
            if frame.time + frame.samples / frame.rate > self.end_time:
                frame, _ = split_audio_frame(frame, self.end_time - frame.time)
            self.graph.push(frame)
            yield from self.graph.pull()
        yield from self.graph.pull(partial=True)
