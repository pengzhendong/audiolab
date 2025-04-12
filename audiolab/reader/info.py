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

from importlib.resources import files
from typing import Any

import av
from humanize import naturalsize
from jinja2 import Environment, FileSystemLoader
from lhotse.utils import Seconds

loader = FileSystemLoader(files("audiolab.reader").joinpath("templates"))
template = Environment(loader=loader).get_template("info.txt")


class Info:
    def __init__(self, file: Any, stream_id: int = 0):
        self.container = av.open(file)
        self.stream = self.container.streams.audio[stream_id]
        self.channels = self.stream.channels
        self.rate = self.stream.rate
        self.sample_rate = self.stream.sample_rate
        self.precision = self.stream.format.bits
        self.bit_rate = self.stream.bit_rate or self.container.bit_rate
        self.metadata = self.stream.metadata

    @property
    def num_seconds(self) -> Seconds:
        return Seconds(self.stream.duration * self.stream.time_base)

    @property
    def num_cdda_sectors(self) -> float:
        return round(self.num_seconds * 75, 2)

    @property
    def num_samples(self) -> int:
        # Number of samples per channel
        return int(self.num_seconds * self.stream.rate)

    @property
    def duration(self):
        hours, rest = divmod(self.num_seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    @staticmethod
    def rstrip_zeros(s: str) -> str:
        return " ".join(x.rstrip("0").rstrip(".") for x in s.split())

    def __str__(self):
        return template.render(
            name=self.container.name,
            channels=self.channels,
            rate=self.rate,
            precision=self.precision,
            duration=Info.rstrip_zeros(self.duration),
            num_samples=self.num_samples,
            num_cdda_sectors=Info.rstrip_zeros(str(self.num_cdda_sectors)),
            size=Info.rstrip_zeros(naturalsize(self.container.size)),
            bit_rate=Info.rstrip_zeros(naturalsize(self.bit_rate).rstrip("B")),
            codec=self.stream.codec.long_name,
            metadata=self.metadata,
        )
