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
from functools import cached_property
from typing import Any

import numpy as np
from humanize import naturalsize

from audiolab.av.typing import Seconds
from audiolab.av.utils import get_template
from audiolab.reader.backend.registry import open_backend


class Info:
    def __init__(
        self,
        source: Any,
        frame_size: int | None = None,
        forced_decoding: bool = False,
        backends: list[str] | None = None,
    ):
        self.source = source
        self.backend = open_backend(source, frame_size, forced_decoding, backends)

    def close(self):
        backend = self.backend
        if backend is None:
            return
        self.backend = None
        backend.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    @cached_property
    def bits_per_sample(self) -> int | None:
        return self.backend.bits_per_sample

    @property
    def bit_rate(self) -> float | int | None:
        return self.backend.bit_rate

    @cached_property
    def codec(self) -> str:
        return self.backend.codec

    @cached_property
    def duration(self) -> Seconds | None:
        return self.backend.duration

    @cached_property
    def dtype(self) -> np.dtype:
        return self.backend.dtype

    @cached_property
    def format(self) -> str:
        return self.backend.format

    @cached_property
    def layout(self) -> str:
        return self.backend.layout

    @cached_property
    def name(self) -> str:
        return self.backend.name

    @property
    def num_channels(self) -> int:
        return self.backend.num_channels

    @property
    def num_frames(self) -> int | None:
        return self.backend.num_frames

    @property
    def metadata(self) -> dict[str, str]:
        return self.backend.metadata

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    @property
    def seekable(self) -> bool:
        return self.backend.seekable

    @property
    def size(self) -> int | None:
        return self.backend.size

    @property
    def cdda_sectors(self) -> float | None:
        if self.duration is None:
            return None
        return round(self.duration * 75, 2)

    @staticmethod
    def rstrip_zeros(s: int | float | str | None) -> str:
        if s is None:
            return "N/A"
        if not isinstance(s, str):
            s = str(s)
        return " ".join(part.rstrip("0").rstrip(".") if "." in part else part for part in s.split())

    @staticmethod
    def format_bit_rate(bit_rate: float | int | None) -> str:
        if bit_rate is None or bit_rate <= 0:
            return "N/A"
        formatted_bit_rate = naturalsize(bit_rate).rstrip("B")
        return Info.rstrip_zeros(formatted_bit_rate) + "bps"

    @staticmethod
    def format_duration(duration: Seconds | None) -> str:
        if duration is None:
            return "N/A"
        hours, rest = divmod(duration, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    @staticmethod
    def format_name(name: str, container_format: str) -> str:
        if name.upper().endswith(container_format.upper()):
            return f"'{name}'"
        if name in ("<none>", "<stdin>"):
            return f"{name} ({container_format})"
        return f"'{name}' ({container_format})"

    @staticmethod
    def format_size(size: int | None) -> str:
        if size not in (-1, -38, -78, None):
            return Info.rstrip_zeros(naturalsize(size))
        return Info.rstrip_zeros(size)

    def __str__(self):
        return get_template("info").render(
            name=Info.format_name(self.name, self.format),
            channels=self.num_channels,
            rate=self.sample_rate,
            precision="N/A" if self.bits_per_sample is None else self.bits_per_sample,
            duration=Info.format_duration(self.duration),
            samples="N/A" if self.num_frames is None else self.num_frames,
            cdda_sectors=Info.rstrip_zeros(self.cdda_sectors),
            size=Info.format_size(self.size),
            bit_rate=Info.format_bit_rate(self.bit_rate),
            codec=self.codec,
            metadata=self.metadata,
        )
