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

from base64 import b64encode
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from IPython.display import Audio
from lhotse import Recording
from lhotse.cut import Cut

from . import filters
from .filters import Filter
from .graph import AudioGraph
from .info import Info
from .reader import Reader
from .stream_reader import StreamReader
from .utils import load_url

UINT32MAX = 2**32 - 1


def load_audio(
    file: Any,
    stream_id: int = 0,
    offset: float = 0.0,
    duration: Optional[float] = None,
    filters: List[Filter] = [],
    format: Optional[str] = None,
    frame_size: Union[int, str] = UINT32MAX,
    return_ndarray: bool = True,
) -> Union[Iterator[Tuple[np.ndarray, int]], Tuple[np.ndarray, int]]:
    reader = Reader(file, stream_id, offset, duration, filters, format, frame_size, return_ndarray)
    generator = reader.__iter__()
    if frame_size < UINT32MAX:
        return generator
    return next(generator)


def encode(
    audio: Union[str, Path, np.ndarray, torch.Tensor, Cut, Recording],
    rate: Optional[int] = None,
    sample_fmt: str = "flt",
    channel_layout: Union[int, str] = "mono",
    make_wav: bool = True,
) -> Tuple[str, int]:
    """Transform an audio to a PCM bytestring"""
    if isinstance(audio, (str, Path)):
        if rate is None:
            aformat = filters.aformat(sample_fmts=sample_fmt, channel_layouts=channel_layout)
        else:
            aformat = filters.aformat(sample_fmts=sample_fmt, sample_rates=rate, channel_layouts=channel_layout)
        audio, rate = load_audio(audio, filters=[aformat])
    elif isinstance(audio, (Cut, Recording)):
        if rate is not None:
            audio = audio.resample(rate)
        rate = audio.sampling_rate
        audio = audio.load_audio()
    if make_wav:
        return Audio(audio, rate=rate).src_attr(), rate
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768
    # float32 to int16 bytes (Do not normalize for streaming chunks)
    audio, _ = Audio._validate_and_normalize_with_numpy(np.clip(audio, -1, 1), False)
    return b64encode(audio).decode("ascii"), rate


def info(file: Any, stream_id: int = 0) -> Info:
    return Info(file, stream_id)


__all__ = ["AudioGraph", "Reader", "StreamReader", "filters", "load_audio", "load_url", "info"]
