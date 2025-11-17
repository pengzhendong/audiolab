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
from typing import Iterator, List, Optional

import av
from av import AudioCodecContext

from audiolab.av import aformat
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFormat, AudioFrame, Dtype, Filter


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
        always_2d: bool = True,
    ):
        """
        Create a StreamReader object.

        Args:
            filters: The filters to apply to the audio stream.
            dtype: The data type of the output audio frames.
            is_planar: Whether the output audio frames are planar.
            format: The format of the output audio frames.
            rate: The sample rate of the output audio frames.
            to_mono: Whether to convert the output audio frames to mono.
            frame_size: The frame size of the audio frames.
            return_ndarray: Whether to return the audio frames as ndarrays.
        """
        self._codec_context = None
        self._graph = None
        self.bytes_io = BytesIO()
        self.bytes_per_decode_attempt = 0
        if not all([dtype is None, format is None, rate is None, to_mono is None]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, rate, to_mono))
        self.filters = filters
        self.frame_size = frame_size
        self.return_ndarray = return_ndarray
        self.always_2d = always_2d
        self.offset = None
        self.packet = None

    @property
    def codec_context(self) -> Optional[AudioCodecContext]:
        if self._codec_context is None:
            if self.packet is None:
                return None
            self._codec_context = self.packet.stream.codec_context
            assert self._codec_context.frame_size in (0, 1)
        return self._codec_context

    @property
    def graph(self) -> Optional[Graph]:
        if self._graph is None:
            if self.packet is None:
                return None
            self._graph = Graph(
                self.packet.stream,
                filters=self.filters,
                frame_size=self.frame_size,
                return_ndarray=self.return_ndarray,
                always_2d=self.always_2d,
            )
        return self._graph

    def push(self, frame: bytes):
        """
        Push a frame of audio data to the audio.

        Args:
            frame: The frame of audio data to push.
        """
        self.bytes_io.write(frame)
        self.bytes_per_decode_attempt += len(frame)

    def pull(self, partial: bool = False) -> Iterator[AudioFrame]:
        """
        Pull an audio frame from the audio.

        Args:
            partial: Whether to pull a partial frame.
        Yields:
            The audio frame.
        """
        if partial or self.bytes_per_decode_attempt * 2 >= self.frame_size:
            self.bytes_per_decode_attempt = 0
            try:
                self.bytes_io.seek(0)
                container = av.open(self.bytes_io, metadata_encoding="latin1")
                for packet in container.demux():
                    self.packet = packet
                    if self.packet.pts is None and not partial:
                        continue
                    # o: current frame
                    # pts: self.offset, frame.pts, packet.pts
                    # +---+---+---+---+---+
                    # | x | x | x | o |   |
                    # +---+---+---+---+---+
                    #             ↑
                    #             pts
                    if self.offset is not None and (self.packet.pts is None or self.offset > self.packet.pts):
                        continue
                    for frame in self.codec_context.decode(packet):
                        self.offset = frame.pts + int(frame.samples / packet.stream.rate / packet.stream.time_base)
                        self.graph.push(frame)
                        yield from self.graph.pull()
                    yield from self.graph.pull(partial=partial)
            except (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
                pass

    def reset(self):
        self._codec_context = None
        self._graph = None
        self.bytes_io = BytesIO()
        self.bytes_per_decode_attempt = 0
        self.offset = None
        self.packet = None
