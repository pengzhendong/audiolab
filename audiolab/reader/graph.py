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
from typing import List

from av import AudioFrame, AudioStream, FFmpegError
from av.filter import Graph

from .filters import Filter
from .utils import to_ndarray


class AudioGraph:
    def __init__(
        self,
        stream: AudioStream,
        filters: List[Filter],
        frame_size: int,
    ):
        self.graph = Graph()
        nodes = [self.graph.add_abuffer(template=stream)]
        for _filter in filters:
            name, args, kwargs = (
                (_filter, None, {}) if isinstance(_filter, str) else ((*_filter, {}) if len(_filter) == 2 else _filter)
            )
            nodes.append(self.graph.add(name, args, **kwargs))
        nodes.append(self.graph.add("abuffersink"))
        self.graph.link_nodes(*nodes).configure()

        frame_size = int(frame_size) if frame_size is not None else 0
        if frame_size > 0:
            self.graph.set_audio_frame_size(frame_size)

    def push(self, frame: AudioFrame):
        self.graph.push(frame)

    def pull(self, partial: bool = False):
        if partial:
            self.graph.push(None)
        while True:
            try:
                frame = self.graph.pull()
                yield to_ndarray(frame), frame.rate
            except EOFError:
                break
            except FFmpegError as e:
                if e.errno != errno.EAGAIN:
                    raise
                break
