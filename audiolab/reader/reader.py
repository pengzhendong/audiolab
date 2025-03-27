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

from typing import Any, Generator, List, Optional, Tuple

import av
from av import AudioFifo, AudioFrame

from .utils import build_graph, to_ndarray


class Reader:
    def __init__(
        self,
        file: Any,
        stream_id: int = 0,
        block_size: int = 1024,
        offset: float = 0.0,
        duration: float = None,
        filters: Optional[List[Tuple[str, str]]] = None,
    ):
        self.container = av.open(file)
        self.stream = self.container.streams.audio[stream_id]
        self.bit_rate = self.stream.bit_rate
        self.channels = self.stream.channels
        self.codec = self.stream.codec_context.codec.name
        self.rate = self.stream.rate

        self.block_size = block_size
        self.start_time = int(offset / self.stream.time_base)
        self.end_time = offset + duration if duration is not None else float("inf")
        if self.start_time > 0:
            self.container.seek(self.start_time, any_frame=True, stream=self.stream)

        self.fifo = AudioFifo()
        self.graph = build_graph(self.stream, filters)

    def resize(self, frame: AudioFrame) -> Generator[AudioFrame]:
        self.fifo.write(frame)
        while self.fifo.samples >= self.block_size:
            yield self.fifo.read(self.block_size)

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            assert frame.time == float(frame.pts * self.stream.time_base)
            if frame.time > self.end_time:
                break
            if frame.pts is not None and self.start_time > 0:
                frame.pts -= self.start_time
            self.graph.push(frame)
            while True:
                try:
                    frame = self.graph.pull()
                except (av.BlockingIOError, av.EOFError):
                    break
                for frame in self.resize(frame):
                    yield to_ndarray(frame), frame.rate

        if self.fifo.samples > 0:
            frame = self.fifo.read(self.fifo.samples, partial=True)
            yield to_ndarray(frame), frame.rate

    def __del__(self):
        self.container.close()
