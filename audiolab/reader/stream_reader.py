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

from io import BytesIO
from typing import List, Optional, Union

import av

from .filters import Filter
from .graph import AudioGraph


class StreamReader:
    def __init__(
        self,
        filters: List[Filter] = [],
        format: Optional[str] = None,
        frame_size: Union[int, str] = 1024,
        return_ndarray: bool = True,
    ):
        self.increment = 0
        self.bytestream = BytesIO()
        self.filters = filters
        self.format = format
        self.frame_size = frame_size
        self.graph = None
        self.offset = 0
        self.return_ndarray = return_ndarray

    def push(self, data: bytes):
        self.increment += len(data)
        self.bytestream.write(data)

    def pull(self, partial: bool = False):
        # Attempt decoding every `self.increment` bytes.
        if self.increment * 2 < self.frame_size and not partial:
            return
        self.increment = 0
        try:
            self.bytestream.seek(0)
            container = av.open(self.bytestream, format=self.format)
            stream = container.streams.audio[0]
            # The bytestream is too short to determine the sample rate.
            if stream.sample_rate == 0:
                return
            if self.graph is None:
                self.graph = AudioGraph(stream, self.filters, self.frame_size, self.return_ndarray)

            container.seek(self.offset, any_frame=True, stream=stream)
            frames = list(container.decode(stream))
            frames = frames[1:] if self.offset > 0 else frames
            frames = frames[:-1] if not partial else frames
            # Overlap frames to avoid discontinuities.
            # +---+---+---+---+
            # |   |   |   | x |
            # +---+---+---+---+
            #         +---+---+---+---+
            #         | x |   |   | x |
            #         +---+---+---+---+
            #                 +---+---+---+---+
            #                 | x |   |   | x |
            #                 +---+---+---+---+
            #                         ↑
            #                         self.offset = frames[:-1][-1].pts
            for frame in frames:
                self.offset = frame.pts
                self.graph.push(frame)
                yield from self.graph.pull()
            yield from self.graph.pull(partial=partial)
        except (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
            pass

    def reset(self):
        self.increment = 0
        self.bytestream = BytesIO()
        self.graph = None
        self.offset = 0
