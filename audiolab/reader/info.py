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

from typing import Any, Union

import av
from av import AudioLayout, AudioStream, Codec, time_base
from av.container import Container
from humanize import naturalsize

from audiolab.av.typing import Seconds
from audiolab.av.utils import get_template, is_streamable


class Info:
    stream: AudioStream
    channels: int
    codec: Codec
    rate: int
    sample_rate: int
    layout: AudioLayout

    def __init__(self, file: Any, stream_id: int = 0, force_decoding: bool = False):
        """
        Create an Info object.

        Args:
            file: The input audio file, path to audio file, bytes of audio data, etc.
            stream_id: The index of the stream to get information from.
            force_decoding: Whether to force decoding the audio file to get the duration.
        """
        self.container = av.open(file)
        self.stream = self.container.streams.audio[stream_id]
        self.channels = self.stream.channels
        self.codec = self.stream.codec
        self.rate = self.stream.rate
        self.sample_rate = self.stream.sample_rate
        self.layout = self.stream.layout
        self.precision = self.stream.format.bits
        self.metadata = {**self.container.metadata, **self.stream.metadata}
        self.is_streamable = is_streamable(self.stream.codec_context)

        # number of audio samples (per channel)
        self.num_samples = None
        self.duration = None
        if self.stream.duration:
            start_time = self.stream.start_time or 0
            self.duration = Seconds((self.stream.duration + start_time) * self.stream.time_base)
            self.num_samples = int(self.duration * self.stream.rate)
        elif self.container.duration:
            start_time = self.container.start_time or 0
            self.duration = Seconds((self.container.duration + start_time) / time_base)
            self.num_samples = int(self.duration * self.stream.rate)
        elif force_decoding:
            # decode the stream to get the duration if the duration is not available
            self.num_samples = 0
            for frame in self.container.decode(self.stream):
                self.num_samples += frame.samples
            self.duration = Seconds(self.num_samples / self.stream.rate)

    @property
    def bit_rate(self) -> Union[int, None]:
        """
        Get the bit rate of the audio stream.

        Returns:
            The bit rate of the audio stream.
        """
        bit_rate = self.stream.bit_rate or self.container.bit_rate
        if bit_rate in (None, 0) and self.duration is not None:
            # bytes * 8 / seconds
            bit_rate = self.container.size * 8 / self.duration
        return bit_rate

    @property
    def num_cdda_sectors(self) -> Union[float, None]:
        """
        Get the number of CDDA sectors of the audio stream.

        Returns:
            The number of CDDA sectors of the audio stream.
        """
        return None if self.duration is None else round(self.duration * 75, 2)

    @staticmethod
    def rstrip_zeros(s: Union[int, float, str]) -> str:
        """
        Remove trailing zeros from a string.

        Args:
            s: The string to remove trailing zeros from.
        Returns:
            The string with trailing zeros removed.
        """
        if not isinstance(s, str):
            s = str(s)
        return " ".join(x.rstrip("0").rstrip(".") for x in s.split())

    @staticmethod
    def format_bit_rate(bit_rate: Union[int, None]) -> str:
        """
        Format the bit rate of the audio stream.

        Args:
            bit_rate: The bit rate of the audio stream.
        Returns:
            The formatted bit rate of the audio stream.
        """
        if bit_rate is None or bit_rate <= 0:
            return "N/A"
        bit_rate = naturalsize(bit_rate).rstrip("B")
        return Info.rstrip_zeros(bit_rate) + "bps"

    @staticmethod
    def format_duration(duration: Union[Seconds, None]) -> str:
        """
        Format the duration of the audio stream.

        Args:
            duration: The duration of the audio stream.
        Returns:
            The formatted duration of the audio stream.
        """
        if duration is None:
            return "N/A"
        hours, rest = divmod(duration, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    @staticmethod
    def format_name(container: Container) -> str:
        """
        Format the name of the container.

        Args:
            container: The container.
        Returns:
            The formatted name of the container.
        """
        name = container.name
        _format_name = container.format.name
        if name.lower().endswith(_format_name.lower()):
            return f"'{name}'"
        # <none> for BytesIO, <stdin> for stdin
        if name in ("<none>", "<stdin>"):
            return f"{name} ({_format_name})"
        return f"'{name}' ({_format_name})"

    @staticmethod
    def format_num_cdda_sectors(num_cdda_sectors: Union[float, None]) -> str:
        """
        Format the number of CDDA sectors of the audio stream.

        Args:
            num_cdda_sectors: The number of CDDA sectors of the audio stream.
        Returns:
            The formatted number of CDDA sectors of the audio stream.
        """
        return "N/A" if num_cdda_sectors is None else Info.rstrip_zeros(num_cdda_sectors)

    @staticmethod
    def format_size(size: int) -> str:
        """
        Format the size of the audio file.

        Args:
            size: The size of the audio file.
        Returns:
            The formatted size of the audio file.
        """
        if size in (-1, -78):
            return "N/A"
        size = naturalsize(size)
        return Info.rstrip_zeros(size)

    def __str__(self):
        """
        Format the audio file information.

        Returns:
            The formatted audio file information.
        """
        return get_template("info").render(
            name=Info.format_name(self.container),
            channels=self.channels,
            rate=self.rate,
            precision=self.precision,
            duration=Info.format_duration(self.duration),
            num_samples=self.num_samples or "N/A",
            num_cdda_sectors=Info.format_num_cdda_sectors(self.num_cdda_sectors),
            size=Info.format_size(self.container.size),
            bit_rate=Info.format_bit_rate(self.bit_rate),
            codec=self.codec.long_name,
            metadata=self.metadata,
        )
