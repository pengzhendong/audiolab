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

import contextlib
from collections.abc import Iterator
from functools import cached_property
from typing import Any
from urllib.parse import urlsplit

import av
from av import time_base
from av.codec import Codec
from av.error import EOFError, InvalidDataError
from av.format import Flags

from audiolab.av import split_audio_frame
from audiolab.av.format import get_dtype
from audiolab.av.typing import UINT32_MAX, Seconds
from audiolab.reader.backend.backend import Backend

NETWORK_TIMEOUT = 10


class PyAV(Backend):
    def __init__(self, source: Any, frame_size: int | None = None, forced_decoding: bool = False):
        super().__init__(source, frame_size, forced_decoding)
        options = {}
        if isinstance(source, str) and urlsplit(source).scheme in {"http", "https"}:
            options["timeout"] = NETWORK_TIMEOUT
        self.container = av.open(source, metadata_errors="ignore", **options)
        self.stream = self.container.streams.audio[0]
        self.dtype = get_dtype(self.stream.format)

    def close(self):
        container = self.container
        if container is None:
            return
        self.container = None
        self.stream = None
        with contextlib.suppress(Exception):
            container.close()
        super().close()

    @cached_property
    def bits_per_sample(self) -> int:
        return self.stream.format.bits

    @cached_property
    def bit_rate(self) -> int | None:
        bit_rate = None
        if self.stream.bit_rate is not None:
            bit_rate = self.stream.bit_rate
        elif self.container.bit_rate is not None:
            bit_rate = self.container.bit_rate
        if bit_rate in (0, None):
            bit_rate = super().bit_rate
        return bit_rate

    @cached_property
    def codec(self) -> Codec:
        return self.stream.codec.long_name

    @cached_property
    def format(self) -> str:
        return self.container.format.name

    @cached_property
    def duration(self) -> Seconds | None:
        if self.forced_decoding:
            num_frames = 0
            try:
                for audio_frame in self.container.decode(self.stream):
                    num_frames += audio_frame.samples
            except (EOFError, InvalidDataError, StopIteration):
                pass
            duration = num_frames / self.stream.rate
        else:
            duration = None
            if self.stream.duration is not None:
                duration = self.stream.duration * self.stream.time_base
            elif self.container.duration is not None:
                duration = self.container.duration / time_base
        return None if duration is None else Seconds(duration)

    @cached_property
    def is_planar(self) -> bool:
        return self.stream.format.is_planar

    @cached_property
    def name(self) -> str:
        return self.container.name

    @cached_property
    def num_channels(self) -> int:
        return self.stream.channels

    @cached_property
    def num_frames(self) -> int | None:
        if self.duration is None:
            return None
        return int(self.duration * self.stream.rate)

    @cached_property
    def metadata(self) -> dict:
        return {**self.container.metadata, **self.stream.metadata}

    @cached_property
    def sample_rate(self) -> int:
        return self.stream.sample_rate

    @cached_property
    def size(self) -> int | None:
        size = super().size
        if size is None:
            size = self.container.size
        return size

    @cached_property
    def seekable(self) -> bool:
        flags = Flags(self.container.format.flags)
        generic_index = Flags.generic_index in flags
        seek_to_pts = Flags.seek_to_pts in flags
        byte_seek = Flags.no_byte_seek not in flags
        return generic_index or seek_to_pts or byte_seek

    def load_audio(self, offset: Seconds = 0, duration: Seconds | None = None) -> Iterator[av.AudioFrame]:
        offset = int(offset / self.stream.time_base)
        self.seek(offset)
        remaining_frames = UINT32_MAX if duration is None else int(duration * self.sample_rate)
        while remaining_frames > 0:
            audio_frame = self.read()
            if audio_frame is None:
                break
            audio_frame = self.split_frame(audio_frame, offset, remaining_frames)
            if audio_frame is None:
                continue
            remaining_frames -= audio_frame.samples
            yield audio_frame

    def read(self) -> av.AudioFrame | None:
        try:
            return next(self.container.decode(self.stream))
        except (EOFError, InvalidDataError, StopIteration):
            return None

    def seek(self, offset: int):
        if offset > 0:
            self.container.seek(offset, stream=self.stream)

    def split_frame(self, audio_frame: av.AudioFrame, offset: int, frames: int) -> av.AudioFrame | None:
        offset = max(offset - audio_frame.pts, 0) * audio_frame.time_base * audio_frame.sample_rate
        _, audio_frame = split_audio_frame(audio_frame, int(offset))
        if audio_frame is not None:
            audio_frame, _ = split_audio_frame(audio_frame, frames)
        return audio_frame
