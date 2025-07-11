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

from enum import Enum
from typing import Dict, Tuple, Union

import bv
import numpy as np


class BaseEnum(Enum):
    """
    The base enum class.
    """

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __getattr__(self, attr):
        return getattr(self.value, attr)


class CodecEnum(BaseEnum):
    """
    The enum for the codec.
    """

    pass


class ContainerFormatEnum(BaseEnum):
    """
    The enum for the container format.
    """

    pass


class AudioFormatEnum(Enum):
    """
    The enum for the audio format.
    """

    pass


class AudioLayoutEnum(BaseEnum):
    """
    The enum for the audio layout.
    """

    pass


Codec = Union[str, bv.Codec]
Dtype = Union[str, type, np.dtype]
Filter = Union[str, Tuple[str, str], Tuple[str, Dict[str, str]], Tuple[str, str, Dict[str, str]]]
AudioFormat = Union[str, bv.AudioFormat]
AudioFrame = Union[np.ndarray, bv.AudioFrame]
AudioLayout = Union[int, str, bv.AudioLayout]
ContainerFormat = Union[str, bv.ContainerFormat]
