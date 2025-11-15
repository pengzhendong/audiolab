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

from typing import Any, Optional, Union

from av.container import Container
from humanize import naturalsize

from audiolab.av.typing import Seconds
from audiolab.av.utils import get_template
from audiolab.backend import Backend, pyav


class Info:
    def __init__(
        self, file: Any, forced_decoding: bool = False, backend: Backend = pyav
    ):
        # ffmpeg -i xx.flac -f wav - | > xx.wav
        # self.is_streamable = is_streamable(self.stream.codec_context)
        self.file = file
        # self.backend = wave(file, forced_decoding)
        # self.backend = soundfile(file, forced_decoding)
        self.backend = backend(file, forced_decoding)

    def __getattr__(self, name):
        return getattr(self.backend, name)

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
    def format_name(container: Container) -> str:
        name = container.name
        _format_name = container.format.name
        if name.lower().endswith(_format_name.lower()):
            return f"'{name}'"
        # <none> for BytesIO, <stdin> for stdin
        if name in ("<none>", "<stdin>"):
            return f"{name} ({_format_name})"
        return f"'{name}' ({_format_name})"

    @staticmethod
    def format_size(size: int) -> str:
        if size not in (-1, -38, -78, None):
            size = naturalsize(size)
        return Info.rstrip_zeros(size)

    def __str__(self):
        return get_template("info").render(
            name=self.file,
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
