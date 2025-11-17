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

from typing import Any, Iterator, Optional, Tuple, Union

import numpy as np

from audiolab.av import aformat
from audiolab.av.graph import Graph
from audiolab.av.typing import UINT32_MAX
from audiolab.backend import Backend
from audiolab.reader.info import Info
from audiolab.reader.reader import Reader
from audiolab.reader.stream_reader import StreamReader


def info(file: Any, forced_decode: bool = False, backend: Optional[Backend] = None) -> Info:
    """
    Get the information of an audio file.

    Args:
        file: The input audio file, audio url, path to audio file, bytes of audio data, etc.
        forced_decode: Whether to forced decoding the audio file to get the duration.
        backend: The backend to use to get the information.
    Returns:
        The information of the audio file.
    """
    return Info(file, forced_decode, backend)


def load_audio(file: Any, **kwargs) -> Union[Iterator[Tuple[np.ndarray, int]], Tuple[np.ndarray, int]]:
    reader = Reader(file, **kwargs)
    if reader.frame_size < UINT32_MAX:
        return reader.__iter__()
    else:
        return reader.load_audio()


__all__ = ["Graph", "Reader", "StreamReader", "aformat", "load_audio"]
