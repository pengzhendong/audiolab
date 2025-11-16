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
from typing import Any, Optional, Union

from av.codec import Codec
from humanize import naturalsize

from audiolab.av.typing import Seconds
from audiolab.av.utils import get_template
from audiolab.backend import Backend, pyav


class Info:
    def __init__(
        self, file: Any, forced_decoding: bool = False, backend: Backend = pyav
    ):
        self.file = file
        # backend = soundfile
        # backend = wave
        self.backend = backend(file, forced_decoding)

    @cached_property
    def bits_per_sample(self) -> int:
        return self.backend.bits_per_sample

    @property
    def bit_rate(self) -> Optional[int]:
        return self.backend.bit_rate

    @cached_property
    def codec(self) -> Union[Codec, str]:
        return self.backend.codec

    @cached_property
    def duration(self) -> Optional[Seconds]:
        return self.backend.duration

    @cached_property
    def format(self) -> str:
        return self.backend.format

    @cached_property
    def name(self) -> str:
        return self.backend.name

    @property
    def num_channels(self) -> int:
        return self.backend.num_channels

    @property
    def num_frames(self) -> int:
        return self.backend.num_frames

    @property
    def metadata(self) -> int:
        return self.backend.metadata

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    @property
    def size(self) -> int:
        return self.backend.size

    @property
    def cdda_sectors(self) -> Optional[float]:
        if self.duration is None:
            return None
        return round(self.duration * 75, 2)

    @property
    def channels(self) -> int:
        return self.num_channels

    @property
    def rate(self) -> int:
        return self.sample_rate

    @property
    def samplerate(self) -> int:
        return self.sample_rate

    @property
    def precision(self) -> int:
        return self.bits_per_sample

    @staticmethod
    def rstrip_zeros(s: Optional[Union[int, float, str]]) -> str:
        if s is None:
            return "N/A"
        if not isinstance(s, str):
            s = str(s)
        return " ".join(x.rstrip("0").rstrip(".") for x in s.split())

    @staticmethod
    def format_bit_rate(bit_rate: Union[int, None]) -> str:
        if bit_rate is None or bit_rate <= 0:
            return "N/A"
        bit_rate = naturalsize(bit_rate).rstrip("B")
        return Info.rstrip_zeros(bit_rate) + "bps"

    @staticmethod
    def format_duration(duration: Union[Seconds, None]) -> str:
        if duration is None:
            return "N/A"
        hours, rest = divmod(duration, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    @staticmethod
    def format_name(name: str, format: str) -> str:
        if name.upper().endswith(format.upper()):
            return f"'{name}'"
        if name in ("<none>", "<stdin>"):
            return f"{name} ({format})"
        return f"'{name}' ({format})"

    @staticmethod
    def format_size(size: int) -> str:
        if size not in (-1, -38, -78, None):
            size = naturalsize(size)
        return Info.rstrip_zeros(size)

    def __str__(self):
        return get_template("info").render(
            name=Info.format_name(self.name, self.format),
            channels=self.channels,
            rate=self.rate,
            precision=self.precision,
            duration=Info.format_duration(self.duration),
            samples="N/A" if self.num_frames is None else self.num_frames,
            cdda_sectors=Info.rstrip_zeros(self.cdda_sectors),
            size=Info.format_size(self.size),
            bit_rate=Info.format_bit_rate(self.bit_rate),
            codec=self.codec if isinstance(self.codec, str) else self.codec.long_name,
            metadata=self.metadata,
        )
