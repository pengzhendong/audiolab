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
import soundfile as sf
from numpy.typing import DTypeLike

from audiolab.av.frame import clip
from audiolab.av.typing import Seconds
from audiolab.reader.backend.backend import FORCED_DECODE_CHUNK_FRAMES, Backend

_subtype_to_bits = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ULAW": 8,
    "ALAW": 8,
    "DWVW_12": 12,
    "DWVW_16": 16,
    "DWVW_24": 24,
    "DPCM_8": 8,
    "DPCM_16": 16,
    "ALAC_16": 16,
    "ALAC_20": 20,
    "ALAC_24": 24,
    "ALAC_32": 32,
}

_subtype_to_dtype = {
    "PCM_S8": np.int8,
    "PCM_U8": np.uint8,
    "PCM_16": np.int16,
    "PCM_24": np.int32,
    "PCM_32": np.int32,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}

_supported_dtypes = (np.int16, np.int32, np.float32, np.float64)


class SoundFile(Backend):
    def __init__(self, source: Any, frame_size: int | None = None, forced_decoding: bool = False):
        super().__init__(source, frame_size, forced_decoding)
        self.sf = sf.SoundFile(source)
        self.output_dtype: DTypeLike | None = None

    def close(self):
        sound_file = self.sf
        if sound_file is None:
            return
        self.sf = None
        with contextlib.suppress(Exception):
            sound_file.close()
        super().close()

    @cached_property
    def bits_per_sample(self) -> int | None:
        return _subtype_to_bits.get(self.sf.subtype)

    @cached_property
    def codec(self) -> str:
        return sf.available_subtypes()[self.sf.subtype]

    @cached_property
    def duration(self) -> Seconds | None:
        if self.num_frames is None:
            return None
        return Seconds(self.num_frames / self.sample_rate)

    @cached_property
    def dtype(self) -> np.dtype:
        return _subtype_to_dtype.get(self.sf.subtype, np.float64)

    @cached_property
    def format(self) -> str:
        return self.sf.format

    @cached_property
    def num_channels(self) -> int:
        return self.sf.channels

    @cached_property
    def num_frames(self) -> int | None:
        if self.forced_decoding:
            num_frames = 0
            pos = self.sf.tell()
            try:
                while True:
                    audio = self.sf.read(FORCED_DECODE_CHUNK_FRAMES)
                    if audio.shape[0] == 0:
                        break
                    num_frames += audio.shape[0]
            except sf.LibsndfileError:
                sound_file = self.sf
                with contextlib.suppress(Exception):
                    sound_file.close()
                seek = getattr(self.source, "seek", None)
                if seek is not None:
                    with contextlib.suppress(OSError, ValueError):
                        seek(0)
                self.sf = sf.SoundFile(self.source)
                num_frames = 0
            self.sf.seek(pos)
        else:
            num_frames = self.sf.frames
            if num_frames >= np.iinfo(np.int32).max:
                num_frames = None
        return num_frames

    @cached_property
    def metadata(self) -> dict:
        return self.sf.copy_metadata()

    @cached_property
    def sample_rate(self) -> int:
        return self.sf.samplerate

    @cached_property
    def seekable(self) -> bool:
        return self.sf.seekable()

    def read(self, nframes: int, dtype: DTypeLike | None = None) -> np.ndarray | None:
        if dtype is None:
            dtype = self.output_dtype if self.output_dtype is not None else self.dtype
        audio = self.sf.read(nframes, dtype=dtype if dtype in _supported_dtypes else np.float64)
        return np.atleast_2d(clip(audio, dtype).T) if audio.shape[0] > 0 else None

    def seek(self, offset: int):
        self.sf.seek(offset)
