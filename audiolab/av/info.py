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

from typing import Any

import bv
from bv import AudioCodecContext, AudioLayout, AudioStream, Codec, time_base
from humanize import naturalsize
from lhotse import Seconds

from audiolab.av.utils import get_template


class Info:
    stream: AudioStream
    channels: int
    codec: Codec
    rate: int
    sample_rate: int
    layout: AudioLayout

    def __init__(self, file: Any, stream_id: int = 0):
        self.container = bv.open(file)
        self.stream = self.container.streams.audio[stream_id]
        self.channels = self.stream.channels
        self.codec = self.stream.codec
        self.rate = self.stream.rate
        self.sample_rate = self.stream.sample_rate
        self.layout = self.stream.layout
        self.precision = self.stream.format.bits
        self.bit_rate = self.stream.bit_rate or self.container.bit_rate
        self.metadata = self.stream.metadata
        self.is_streamable = Info.is_streamable(self.stream.codec_context)

    @staticmethod
    def is_streamable(codec_context: AudioCodecContext) -> bool:
        # https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/avcodec.h#L1045-L1051
        """
        Each submitted frame except the last must contain exactly frame_size samples per channel.
        May be 0 when the codec has AV_CODEC_CAP_VARIABLE_FRAME_SIZE set, then the frame size is not restricted.
        """
        return codec_context.frame_size in (0, 1)

    @property
    def duration(self) -> Seconds:
        if self.stream.duration:
            start_time = self.stream.start_time or 0
            return Seconds((self.stream.duration + start_time) * self.stream.time_base)
        start_time = self.container.start_time or 0
        return Seconds((self.container.duration + start_time) / time_base)

    @property
    def num_cdda_sectors(self) -> float:
        return round(self.duration * 75, 2)

    @property
    def num_samples(self) -> int:
        # Number of samples per channel
        return int(self.duration * self.stream.rate)

    @staticmethod
    def rstrip_zeros(s: str) -> str:
        return " ".join(x.rstrip("0").rstrip(".") for x in s.split())

    def __str__(self):
        hours, rest = divmod(self.duration, 3600)
        minutes, seconds = divmod(rest, 60)
        duration = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
        return get_template("info").render(
            name=self.container.name,
            channels=self.channels,
            rate=self.rate,
            precision=self.precision,
            duration=Info.rstrip_zeros(duration),
            num_samples=self.num_samples,
            num_cdda_sectors=Info.rstrip_zeros(str(self.num_cdda_sectors)),
            size=Info.rstrip_zeros(naturalsize(self.container.size)),
            bit_rate=Info.rstrip_zeros(naturalsize(self.bit_rate).rstrip("B")),
            codec=self.codec.long_name,
            metadata=self.metadata,
        )
