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

import io
import os
from io import BytesIO
from typing import Any, Optional, Union

from av.codec import Codec

from audiolab.av.typing import Seconds


class Backend:
    def __init__(self, file: Any, forced_encoding: bool = False):
        pass

    def seek(self, offset: int, whence: int = io.SEEK_SET):
        pass

    @property
    def size(self) -> Optional[int]:
        if isinstance(self.file, str):
            if os.path.exists(self.file):
                return os.stat(self.file).st_size
        elif isinstance(self.file, BytesIO):
            return len(self.file.getbuffer())
        return None

    @property
    def seekable(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        pass

    @property
    def codec(self) -> Union[Codec, str]:
        pass

    @property
    def duration(self) -> Optional[Seconds]:
        pass

    @property
    def num_frames(self) -> int:
        pass

    @property
    def num_channels(self) -> int:
        pass

    @property
    def bits_per_sample(self) -> int:
        pass

    @property
    def bit_rate(self) -> Optional[int]:
        bit_rate = None
        if self.size is not None:
            if self.duration is not None and self.duration > 0:
                bit_rate = self.size * 8 / self.duration
        return bit_rate

    @property
    def format_name(self):
        return "wav"

    @property
    def metadata(self) -> dict:
        return {}

    @property
    def rate(self) -> int:
        return self.sample_rate

    @property
    def samplerate(self) -> int:
        return self.sample_rate

    @property
    def cdda_sectors(self) -> Optional[float]:
        if self.duration is None:
            return None
        return round(self.duration * 75, 2)

    @property
    def num_cdda_sectors(self) -> Optional[float]:
        return self.cdda_sectors

    @property
    def num_samples(self) -> int:
        return self.num_frames

    @property
    def samples_per_channel(self) -> int:
        return self.num_frames

    @property
    def channels(self) -> int:
        return self.num_channels

    @property
    def precision(self) -> int:
        return self.bits_per_sample
