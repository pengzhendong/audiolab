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

from collections.abc import Iterator
from fractions import Fraction
from io import BytesIO

import av
from numpy.typing import DTypeLike

from audiolab.av import build_filter_chain
from audiolab.av.frame import split_audio_frame
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFormatLike, DecodedChunk, FilterSpec


class StreamReader:
    """Incrementally decode encoded audio bytes.

    Each call to :meth:`pull` retries the complete buffered container and emits
    only frames that have not been returned before. Call ``pull(partial=True)``
    once when no more bytes will be pushed.
    """

    def __init__(
        self,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        is_planar: bool = False,
        sample_format: AudioFormatLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        frame_size: int = 1024,
    ):
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.filters = build_filter_chain(
            filters,
            dtype=dtype,
            is_planar=is_planar,
            sample_format=sample_format,
            sample_rate=sample_rate,
            to_mono=to_mono,
        )
        self.frame_size = frame_size
        self._graph: Graph | None = None
        self._buffer = BytesIO()
        self._new_bytes = 0
        self._next_pts: int | None = None
        self._finalized = False

    def push(self, data: bytes) -> None:
        if self._finalized:
            raise RuntimeError("Cannot push data after the stream has been finalized")
        self._buffer.seek(0, 2)
        self._buffer.write(data)
        self._new_bytes += len(data)

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        if self._finalized or (self._new_bytes == 0 and not partial):
            return

        self._new_bytes = 0
        container = None
        try:
            self._buffer.seek(0)
            container = av.open(self._buffer, metadata_encoding="latin1")
            stream = container.streams.audio[0]
            if self._graph is None:
                self._graph = Graph(stream, filters=self.filters, frame_size=self.frame_size)

            for packet in container.demux(stream):
                for audio_frame in packet.decode():
                    audio_frame = self._remove_decoded_prefix(audio_frame)
                    if audio_frame is None:
                        continue
                    self._graph.push(audio_frame)
                    yield from self._graph.pull()
        except (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
            pass
        finally:
            if container is not None:
                container.close()

        if partial:
            self._finalized = True
            if self._graph is not None:
                yield from self._graph.pull(partial=True)

    def _remove_decoded_prefix(self, audio_frame: av.AudioFrame) -> av.AudioFrame | None:
        if audio_frame.pts is None or audio_frame.time_base is None:
            return audio_frame

        duration_pts = Fraction(audio_frame.samples, audio_frame.rate) / audio_frame.time_base
        frame_end = audio_frame.pts + int(duration_pts)
        if self._next_pts is not None:
            if frame_end <= self._next_pts:
                return None
            if audio_frame.pts < self._next_pts:
                overlap = int(Fraction(self._next_pts - audio_frame.pts) * audio_frame.time_base * audio_frame.rate)
                _, audio_frame = split_audio_frame(audio_frame, overlap)
                if audio_frame is None:
                    return None
        self._next_pts = frame_end
        return audio_frame

    def reset(self) -> None:
        self._graph = None
        self._buffer = BytesIO()
        self._new_bytes = 0
        self._next_pts = None
        self._finalized = False
