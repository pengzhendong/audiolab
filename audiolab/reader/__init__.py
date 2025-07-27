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

from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

from audiolab.av.typing import AudioFormat, Dtype, Filter, Seconds
from audiolab.reader.info import Info
from audiolab.reader.reader import Reader
from audiolab.reader.stream_reader import StreamReader


def info(file: Any, stream_id: int = 0, force_decode: bool = False) -> Info:
    """
    Get the information of an audio file.

    Args:
        file: The input audio file, path to audio file, bytes of audio data, etc.
        stream_id: The index of the stream to get information from.
        force_decode: Whether to force decoding the audio file to get the duration.
    Returns:
        The information of the audio file.
    """
    return Info(file, stream_id, force_decode)


def load_audio(
    file: Any,
    stream_id: int = 0,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    filters: Optional[List[Filter]] = None,
    dtype: Optional[Dtype] = None,
    is_planar: bool = False,
    format: Optional[AudioFormat] = None,
    rate: Optional[int] = None,
    to_mono: bool = False,
    frame_size: Optional[int] = None,
    frame_size_ms: Optional[int] = None,
    return_ndarray: bool = True,
    cache_url: bool = True,
) -> Union[Iterator[Tuple[np.ndarray, int]], Tuple[np.ndarray, int]]:
    """
    Load an audio file.

    Args:
        file: The audio file, path to audio file, bytes of audio data, etc.
        stream_id: The index of the stream to load.
        offset: The offset of the audio stream to load.
        duration: The duration of the audio stream to load.
        filters: The filters to apply to the audio stream.
        dtype: The data type of the audio frames.
        is_planar: Whether the audio frames are planar.
        format: The format of the audio frames.
        rate: The sample rate of the audio frames.
        to_mono: Whether to convert the audio frames to mono.
        frame_size: The frame size of the audio frames.
        frame_size_ms: The frame size in milliseconds of the audio frames.
        return_ndarray: Whether to return the audio frames as ndarrays.
        cache_url: Whether to cache the audio file.
    Returns:
        The audio frames and the sample rate of the audio stream.
    """
    reader = Reader(
        file,
        stream_id,
        offset,
        duration,
        filters,
        dtype,
        is_planar,
        format,
        rate,
        to_mono,
        frame_size,
        frame_size_ms,
        return_ndarray,
        cache_url,
    )
    generator = reader.__iter__()
    if reader.frame_size < np.iinfo(np.uint32).max:
        return generator
    return next(generator)


__all__ = ["Reader", "StreamReader", "load_audio"]
