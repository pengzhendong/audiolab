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

from typing import Any, List, Optional, Union

import av
from lhotse import Seconds

from .filters import Filter
from .graph import AudioGraph
from .utils import load_url, split_audio_frame


class Reader:
    def __init__(
        self,
        file: Any,
        stream_id: int = 0,
        offset: Seconds = 0.0,
        duration: Seconds = None,
        filters: List[Filter] = [],
        format: Optional[str] = None,
        frame_size: Union[int, str] = None,
        return_ndarray: bool = True,
    ):
        # Open and seek url by ffmpeg may cause some issues.
        if isinstance(file, str) and "://" in file:
            file = load_url(file)

        self.container = av.open(file, format=format)
        self.stream = self.container.streams.audio[stream_id]
        self.start_time = int(offset / self.stream.time_base)
        self.end_time = offset + duration if duration is not None else Seconds("inf")
        if self.start_time > 0:
            self.container.seek(self.start_time, any_frame=True, stream=self.stream)
        self.graph = AudioGraph(self.stream, filters, frame_size, return_ndarray)

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

    def __del__(self):
        self.container.close()
