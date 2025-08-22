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

from typing import Any, Dict, Optional

import av

from audiolab.av.format import get_format_dtype
from audiolab.av.frame import to_ndarray
from audiolab.av.typing import AudioFormat, AudioFrame, AudioLayout, Codec, ContainerFormat, Dtype
from audiolab.writer.writer import Writer


def save_audio(
    file: Any,
    frame: AudioFrame,
    rate: int,
    codec: Optional[Codec] = None,
    channels: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    is_planar: Optional[bool] = None,
    format: Optional[AudioFormat] = None,
    layout: Optional[AudioLayout] = None,
    container_format: Optional[ContainerFormat] = None,
    options: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    Save an audio frame to a file.

    Args:
        file: The audio file, path to audio file, bytes, etc.
        rate: The sample rate of the audio stream.
        codec: The codec of the audio container.
        channels: The number of channels of the audio stream.
        dtype: The data type of the audio stream.
        is_planar: Whether the audio stream is planar.
        format: The format of the audio stream.
        layout: The layout of the audio stream.
        container_format: The format of the audio container.
        options: The options of the audio stream.
        **kwargs: Additional arguments for the Writer class.
    """
    if isinstance(frame, av.AudioFrame):
        if format is None:
            dtype = dtype or get_format_dtype(frame.format)
            is_planar = is_planar or frame.format.is_planar
        frame = to_ndarray(frame)

    channels = 1 if frame.ndim == 1 else frame.shape[0]
    assert channels in (1, 2)
    writer = Writer(file, rate, codec, channels, dtype, is_planar, format, layout, container_format, options, **kwargs)
    writer.write(frame)
    writer.close()


__all__ = ["Writer", "save_audio"]
