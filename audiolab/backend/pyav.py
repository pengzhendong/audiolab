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
from typing import Any, Iterator, Optional

import av
import numpy as np
from av import time_base
from av.codec import Codec

from audiolab.av import split_audio_frame
from audiolab.av.format import get_dtype
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFrame, Seconds
from audiolab.backend.backend import Backend


class PyAV(Backend):
    def __init__(
        self,
        file: Any,
        frame_size: Optional[int] = None,
        frame_size_ms: Optional[int] = None,
        return_ndarray: bool = True,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
        cache_url: bool = False,
        forced_decoding: bool = False,
    ):
        super().__init__(file, frame_size, frame_size_ms, always_2d, fill_value, cache_url, forced_decoding)
        self.container = av.open(file, metadata_encoding="latin1")
        self.stream = self.container.streams.audio[0]
        self.return_ndarray = return_ndarray
        self.graph = Graph(
            rate=self.sample_rate,
            dtype=self.dtype,
            is_planar=self.is_planar,
            channels=self.num_channels,
            frame_size=frame_size,
            return_ndarray=return_ndarray,
            fill_value=fill_value,
        )

    @cached_property
    def bits_per_sample(self) -> int:
        return self.stream.format.bits

    @cached_property
    def bit_rate(self) -> Optional[int]:
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
    def dtype(self) -> np.dtype:
        return get_dtype(self.stream.format)

    @cached_property
    def format(self) -> str:
        return self.container.format.name

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
    def is_planar(self) -> bool:
        return self.stream.format.is_planar

    @cached_property
    def name(self) -> str:
        return self.container.name

    @cached_property
    def num_channels(self) -> int:
        return self.stream.channels

    @cached_property
    def num_frames(self) -> Optional[int]:
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
    def size(self) -> Optional[int]:
        size = super().size
        if size is None:
            size = self.container.size
        return size

    @cached_property
    def seekable(self) -> bool:
        # https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/avcodec.h#L1041-L1051
        # Each submitted frame except the last must contain exactly frame_size samples per channel.
        # May be 0 when the codec has AV_CODEC_CAP_VARIABLE_FRAME_SIZE set, then the frame size is not restricted.
        return self.stream.codec_context.frame_size in (0, 1)

    def load_audio(self, offset: Seconds = 0, duration: Optional[Seconds] = None) -> Iterator[AudioFrame]:
        self.seek(int(offset * self.sample_rate))
        end_time = self.duration if duration is None else min(self.duration, offset + duration)
        for frame in self.container.decode(self.stream):
            assert frame.time == float(frame.pts * self.stream.time_base)
            if frame.time > end_time:
                break
            offset = int(round(end_time - frame.time, 5) * frame.sample_rate)
            if offset < frame.samples:
                frame, _ = split_audio_frame(frame, offset)

            self.graph.push(frame)
            for frame in self.graph.pull():
                yield frame[0] if self.return_ndarray else frame
        for frame in self.graph.pull(partial=True):
            yield frame[0] if self.return_ndarray else frame

    def read(self, nframes: int) -> np.ndarray:
        raise NotImplementedError

    def seek(self, offset: int):
        self.container.seek(offset, any_frame=True, stream=self.stream)
