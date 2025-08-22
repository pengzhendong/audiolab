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

import logging
from io import BytesIO
from typing import Any, Dict, Optional

import av
import numpy as np

from audiolab.av import from_ndarray
from audiolab.av.format import get_format
from audiolab.av.typing import AudioFormat, AudioFrame, AudioLayout, Codec, ContainerFormat, Dtype

logger = logging.getLogger(__name__)


class Writer:
    def __init__(
        self,
        file: Any,
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
        Create a Writer object.

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
        Returns:
            The Writer object.
        """
        if isinstance(file, BytesIO):
            assert container_format is not None
            if isinstance(container_format, av.ContainerFormat):
                container_format = container_format.name
            self.container = av.open(file, "w", container_format)
        else:
            self.container = av.open(file, "w")
        # set and check codec
        codec = codec or self.container.default_audio_codec
        if isinstance(codec, str):
            codec = av.Codec(codec, "w")
        assert codec.name in self.container.supported_codecs
        # set and check format
        if dtype is not None:
            format = get_format(dtype, is_planar, codec.audio_formats)
        else:
            format = format or codec.audio_formats[0]
        if isinstance(format, av.AudioFormat):
            format = format.name
        assert format in set(format.name for format in codec.audio_formats)
        if layout is None:
            assert channels in (1, 2)
            layout = "mono" if channels == 1 else "stereo"

        kwargs = {**kwargs, **{"format": format, "layout": layout}}
        self.stream = self.container.add_stream(codec.name, rate, options, **kwargs)

    def write(self, frame: AudioFrame):
        """
        Write an audio frame to the audio stream.

        Args:
            frame: The audio frame to write.
        """
        if isinstance(frame, np.ndarray):
            frame = np.atleast_2d(frame)
            assert frame.ndim == 2, "Audio frame must be 1D (samples,) or 2D (channels, samples)"
            assert frame.shape[0] == self.stream.channels, "Number of channels in frame does not match stream"
            frame = from_ndarray(frame, self.stream.format.name, self.stream.layout, self.stream.rate)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        """
        Close the audio stream.
        """
        self.container.close()
