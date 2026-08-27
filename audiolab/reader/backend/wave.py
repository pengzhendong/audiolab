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
import wave
from functools import cached_property
from typing import Any

import numpy as np
from av.codec import Codec

from audiolab.av.typing import Seconds
from audiolab.reader.backend.backend import FORCED_DECODE_CHUNK_FRAMES, Backend

_bits_to_codec = {8: "pcm_u8le", 16: "pcm_s16le", 24: "pcm_s32le", 32: "pcm_s32le"}
_bits_to_dtype = {8: np.uint8, 16: np.int16, 24: np.int32, 32: np.int32}


class Wave(Backend):
    def __init__(self, source: Any, frame_size: int | None = None, forced_decoding: bool = False):
        super().__init__(source, frame_size, forced_decoding)
        self.wave = wave.open(source)

    def close(self):
        wave_file = self.wave
        if wave_file is None:
            return
        self.wave = None
        with contextlib.suppress(Exception):
            wave_file.close()
        super().close()

    @cached_property
    def bits_per_sample(self) -> int:
        return self.wave.getsampwidth() * 8

    @cached_property
    def codec(self) -> str:
        return Codec(_bits_to_codec[self.bits_per_sample]).long_name

    @cached_property
    def duration(self) -> Seconds | None:
        if self.num_frames is None:
            return None
        return Seconds(self.num_frames / self.sample_rate)

    @cached_property
    def dtype(self) -> np.dtype:
        return _bits_to_dtype[self.bits_per_sample]

    @cached_property
    def format(self) -> str:
        return "WAV"

    @cached_property
    def num_channels(self) -> int:
        return self.wave.getnchannels()

    @cached_property
    def num_frames(self) -> int | None:
        if self.forced_decoding:
            position = self.wave.tell()
            num_frames = 0
            while buffer := self.wave.readframes(FORCED_DECODE_CHUNK_FRAMES):
                num_frames += len(buffer) // (self.wave.getnchannels() * self.wave.getsampwidth())
            self.wave.setpos(position)
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
            audio = np.frombuffer(buffer, np.uint8)
            audio = (
                (audio[2::3].astype(np.int32) << 16)
                | (audio[1::3].astype(np.int32) << 8)
                | audio[0::3].astype(np.int32)
            )
            audio[audio > 0x7FFFFF] -= 0x1000000
        else:
            audio = np.frombuffer(buffer, self.dtype)
        return audio.reshape(-1, self.num_channels).T

    def read(self, nframes: int) -> np.ndarray | None:
        buffer = self.wave.readframes(nframes)
        return self.frombuffer(buffer) if len(buffer) > 0 else None

    def seek(self, offset: int):
        if offset > 0:
            self.wave.setpos(offset)
