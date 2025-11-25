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

from typing import Any, Optional, Tuple

import av
import numpy as np
from av.codec.codec import UnknownCodecError

from audiolab.av import from_ndarray
from audiolab.av.format import dtype_formats
from audiolab.av.frame import clip
from audiolab.av.layout import standard_channel_layouts
from audiolab.av.typing import ContainerFormat, Dtype
from audiolab.writer.backend.backend import Backend


class PyAV(Backend):
    def __init__(self, file: Any, sample_rate: int, dtype: Optional[Dtype] = None, format: ContainerFormat = "WAV"):
        super().__init__(file, sample_rate, dtype, format)
        self.container = av.open(self.file, "w", format=self.format)
        self.num_channels = None
        self.stream = None

    def open(self):
        kwargs = {"layout": standard_channel_layouts[self.num_channels][0]}
        audio_codec, audio_format = self.guess_codec_format()
        if audio_format is not None:
            kwargs["format"] = audio_format
        self.stream = self.container.add_stream(audio_codec, self.sample_rate, **kwargs)

    def guess_codec_format(self) -> Tuple[str, str]:
        default_codec = self.container.default_audio_codec
        if self.dtype is None:
            return default_codec, None
        else:
            dtype_format = dtype_formats[self.dtype]
            for audio_format in av.Codec(default_codec, "w").audio_formats:
                if audio_format.name.startswith(dtype_format):
                    return default_codec, audio_format.name

            supported_codecs = self.container.supported_codecs
            codecs = sorted(supported_codecs, key=lambda x: (not x.startswith("pcm_") or x.endswith("law"), x))
            for codec in codecs:
                try:
                    audio_formats = av.Codec(codec, "w").audio_formats
                    if audio_formats is None:
                        continue
                    for audio_format in audio_formats:
                        if audio_format.name.startswith(dtype_format):
                            return codec, audio_format.name
                except UnknownCodecError:
                    pass

    def write(self, frame: np.ndarray):
        if self.dtype is None:
            self.dtype = frame.dtype
        frame = np.atleast_2d(clip(frame, self.dtype))
        if self.num_channels is None:
            self.num_channels = frame.shape[0]
        if self.stream is None:
            self.open()
        frame = from_ndarray(frame, self.stream.format.name, self.stream.layout, self.stream.rate)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if not self.is_closed:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            except ValueError:
                pass
            self.container.close()
            super().close()
