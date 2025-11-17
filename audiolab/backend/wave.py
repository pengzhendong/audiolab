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

import wave
from functools import cached_property
from typing import Any, Optional

import numpy as np
from av.codec import Codec

from audiolab.av.typing import Seconds
from audiolab.backend.backend import Backend

_bits_to_codec_map = {8: "pcm_u8le", 16: "pcm_s16le", 24: "pcm_s32le", 32: "pcm_s32le"}
_bits_to_dtype_map = {8: np.uint8, 16: np.int16, 24: np.int32, 32: np.int32}


class Wave(Backend):
    def __init__(
        self,
        file: Any,
        frame_size: Optional[int] = None,
        frame_size_ms: Optional[int] = None,
        always_2d: bool = True,
        fill_value: Optional[float] = None,
        cache_url: bool = False,
        forced_decoding: bool = False,
        **kwargs,
    ):
        super().__init__(file, frame_size, frame_size_ms, always_2d, fill_value, cache_url, forced_decoding)
        self.wave = wave.open(file)

    @cached_property
    def bits_per_sample(self) -> int:
        return self.wave.getsampwidth() * 8

    @cached_property
    def codec(self) -> str:
        return Codec(_bits_to_codec_map[self.bits_per_sample]).long_name

    @cached_property
    def duration(self) -> Optional[Seconds]:
        if self.num_frames is None:
            return None
        return Seconds(self.num_frames / self.sample_rate)

    @cached_property
    def dtype(self) -> np.dtype:
        return _bits_to_dtype_map[self.bits_per_sample]

    @cached_property
    def format(self) -> str:
        return "WAV"

    @cached_property
    def num_channels(self) -> int:
        return self.wave.getnchannels()

    @cached_property
    def num_frames(self) -> Optional[int]:
        if self.forced_decoding:
            num_frames = self.read(np.iinfo(np.int32).max).shape[0]
            self.wave.rewind()
        else:
            num_frames = self.wave.getnframes()
            if num_frames >= np.iinfo(np.int32).max:
                num_frames = None
        return num_frames

    @cached_property
    def sample_rate(self) -> int:
        return self.wave.getframerate()

    @cached_property
    def seekable(self) -> bool:
        return True

    def frombuffer(self, buffer: bytes) -> np.ndarray:
        if self.bits_per_sample == 24:
            frames = np.frombuffer(buffer, np.uint8)
            frames = (
                (frames[2::3].astype(np.int32) << 16)
                | (frames[1::3].astype(np.int32) << 8)
                | frames[0::3].astype(np.int32)
            )
            frames[frames > 0x7FFFFF] -= 0x1000000
        else:
            frames = np.frombuffer(buffer, self.dtype)
        return frames.reshape(-1, self.num_channels).T

    def read(self, nframes: int) -> np.ndarray:
        buffer = self.wave.readframes(nframes)
        return self.frombuffer(buffer)

    def seek(self, offset: int):
        self.wave.setpos(offset)
