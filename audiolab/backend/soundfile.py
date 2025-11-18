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
from typing import Any, Optional

import numpy as np
import soundfile as sf

from audiolab.av.typing import Dtype, Seconds
from audiolab.backend.backend import Backend

_subtype_to_bits_map = {
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

_subtype_to_dtype_map = {
    "PCM_S8": np.int8,
    "PCM_U8": np.uint8,
    "PCM_16": np.int16,
    "PCM_24": np.int32,
    "PCM_32": np.int32,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}


class SoundFile(Backend):
    def __init__(self, file: Any, frame_size: Optional[int] = None, forced_decoding: bool = False):
        super().__init__(file, frame_size, forced_decoding)
        self.sf = sf.SoundFile(file)

    @cached_property
    def bits_per_sample(self) -> Optional[int]:
        return _subtype_to_bits_map.get(self.sf.subtype, None)

    @cached_property
    def codec(self) -> str:
        return sf.available_subtypes()[self.sf.subtype]

    @cached_property
    def duration(self) -> Optional[Seconds]:
        if self.num_frames is None:
            return None
        return Seconds(self.num_frames / self.sample_rate)

    @cached_property
    def dtype(self) -> np.dtype:
        return _subtype_to_dtype_map.get(self.sf.subtype, np.float32)

    @cached_property
    def format(self) -> str:
        return self.sf.format

    @cached_property
    def num_channels(self) -> int:
        return self.sf.channels

    @cached_property
    def num_frames(self) -> Optional[int]:
        if self.forced_decoding:
            num_frames = 0
            pos = self.sf.tell()
            try:
                frames = self.sf.read()
                num_frames = frames.shape[0]
            except sf.LibsndfileError:
                self.sf = sf.SoundFile(self.file)
            self.seek(pos)
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

    def read(self, nframes: int, dtype: Optional[Dtype] = None) -> Optional[np.ndarray]:
        if dtype is None:
            dtype = self.dtype
        frames = self.sf.read(nframes, dtype=dtype)
        return np.atleast_2d(frames.T) if frames.shape[0] > 0 else None

    def seek(self, offset: int):
        self.sf.seek(offset)
