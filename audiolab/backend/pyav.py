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

from functools import cached_property
from typing import Any, Optional

import av
from av import time_base
from av.codec import Codec

from audiolab.av.typing import Seconds
from audiolab.backend.backend import Backend


class PyAV(Backend):
    def __init__(self, file: Any, forced_decoding: bool = False):
        self.container = av.open(file, metadata_encoding="latin1")
        self.stream = self.container.streams.audio[0]
        self.forced_decoding = forced_decoding

    def __getattr__(self, name):
        if hasattr(self.stream, name):
            return getattr(self.stream, name)
        return getattr(self.container, name)

    @property
    def size(self) -> Optional[int]:
        size = super().size
        if size is None:
            size = self.container.size
        return size

    @property
    def sample_rate(self) -> int:
        return self.stream.sample_rate

    @cached_property
    def duration(self) -> Optional[Seconds]:
        if self.forced_decoding:
            num_frames = 0
            for frame in self.container.decode(self.stream):
                num_frames += frame.samples
            duration = num_frames / self.stream.rate
        else:
            duration = None
            if self.stream.duration is not None:
                duration = self.stream.duration * self.stream.time_base
            elif self.container.duration is not None:
                duration = self.container.duration / time_base
        return None if duration is None else Seconds(duration)

    @cached_property
    def num_frames(self) -> Optional[int]:
        if self.duration is None:
            return None
        return int(self.duration * self.stream.rate)

    @property
    def num_channels(self) -> int:
        return self.stream.channels

    @property
    def bits_per_sample(self) -> int:
        return self.stream.format.bits

    @property
    def bit_rate(self) -> Optional[int]:
        bit_rate = None
        if self.stream.bit_rate is not None:
            bit_rate = self.stream.bit_rate
        elif self.container.bit_rate is not None:
            bit_rate = self.container.bit_rate
        if bit_rate in (0, None):
            bit_rate = super().bit_rate
        return bit_rate

    @property
    def codec(self) -> Codec:
        return self.stream.codec

    @property
    def metadata(self) -> dict:
        return {**self.container.metadata, **self.stream.metadata}
