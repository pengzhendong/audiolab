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
from typing import List, Optional

import bv

from audiolab.av import AudioGraph, Info, aformat
from audiolab.av.typing import AudioFormat, Dtype, Filter


class StreamReader:
    def __init__(
        self,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = 1024,
        return_ndarray: bool = True,
    ):
        self._codec_context = None
        self._graph = None
        self.bytestream = BytesIO()
        self.bytes_per_decode_attempt = 0
        if not all([dtype is None, format is None, rate is None, to_mono is None]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, rate, to_mono))
        self.filters = filters
        self.frame_size = frame_size
        self.offset = None
        self.return_ndarray = return_ndarray
        self.packet = None

    @property
    def codec_context(self):
        if self._codec_context is None:
            if self.packet is None:
                return None
            self._codec_context = self.packet.stream.codec_context
            assert Info.is_streamable(self._codec_context), "Only support streamable codec"
        return self._codec_context

    @property
    def graph(self):
        if self._graph is None:
            if self.packet is None:
                return None
            self._graph = AudioGraph(
                stream=self.packet.stream,
                filters=self.filters,
                frame_size=self.frame_size,
                return_ndarray=self.return_ndarray,
            )
        return self._graph

    @property
    def is_decoded(self):
        # x: decoded frames
        # o: current frame
        # pts: self.offset, frame.pts, packet.pts
        # +---+---+---+---+---+
        # | x | x | x | o |   |
        # +---+---+---+---+---+
        #             ↑
        #             pts
        if self.offset is None:
            return False
        if self.packet.pts is None:
            return False
        if self.offset <= self.packet.pts:
            return False
        return True

    def should_decode(self, partial: bool = False):
        if partial:
            return True
        if self.bytes_per_decode_attempt * 2 < self.frame_size:
            return False
        self.bytes_per_decode_attempt = 0
        return True

    def ready_for_decode(self, partial: bool = False):
        if self.packet.pts is None and not partial:
            return False
        return not self.is_decoded

    def push(self, chunk: bytes):
        self.bytestream.write(chunk)
        self.bytes_per_decode_attempt += len(chunk)

    def pull(self, partial: bool = False):
        if not self.should_decode(partial):
            return
        try:
            self.bytestream.seek(0)
            container = bv.open(self.bytestream)
            for packet in container.demux():
                self.packet = packet
                if not self.ready_for_decode(partial):
                    continue
                for frame in self.codec_context.decode(packet):
                    self.offset = frame.pts + int(frame.samples / packet.stream.rate / packet.stream.time_base)
                    self.graph.push(frame)
                    yield from self.graph.pull()
                yield from self.graph.pull(partial=partial)
        except (bv.EOFError, bv.InvalidDataError, bv.OSError, bv.PermissionError):
            pass

    def reset(self):
        self._codec_context = None
        self._graph = None
        self.bytestream = BytesIO()
        self.bytes_per_decode_attempt = 0
        self.offset = None
        self.packet = None
