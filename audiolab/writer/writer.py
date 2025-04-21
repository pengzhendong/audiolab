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

import bv
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
        if isinstance(file, BytesIO):
            assert container_format is not None
            if isinstance(container_format, bv.ContainerFormat):
                container_format = container_format.name
            self.container = bv.open(file, "w", container_format)
        else:
            self.container = bv.open(file, "w")
        # set and check codec
        codec = codec or self.container.default_audio_codec
        if isinstance(codec, str):
            codec = bv.Codec(codec, "w")
        assert codec.name in self.container.supported_codecs
        # set and check format
        if dtype is not None:
            format = get_format(dtype, is_planar, codec.audio_formats)
        else:
            format = format or codec.audio_formats[0]
        if isinstance(format, bv.AudioFormat):
            format = format.name
        assert format in set(format.name for format in codec.audio_formats)
        if layout is None:
            assert channels in (1, 2)
            layout = "mono" if channels == 1 else "stereo"

        kwargs = {**kwargs, **{"format": format, "layout": layout}}
        self.stream = self.container.add_stream(codec.name, rate, options, **kwargs)

    def write(self, frame: AudioFrame):
        if isinstance(frame, np.ndarray):
            frame = from_ndarray(frame, self.stream.format.name, self.stream.layout, self.stream.rate)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        self.container.close()
