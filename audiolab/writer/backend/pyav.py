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

from typing import Any

import av
import numpy as np
from av.codec.codec import UnknownCodecError
from numpy.typing import DTypeLike

from audiolab.av import from_ndarray
from audiolab.av.format import dtype_formats
from audiolab.av.layout import standard_channel_layouts
from audiolab.av.typing import ContainerFormatLike
from audiolab.writer.backend.backend import Backend


class PyAV(Backend):
    def __init__(
        self,
        destination: Any,
        sample_rate: int,
        dtype: DTypeLike | None = None,
        container_format: ContainerFormatLike = "WAV",
    ):
        super().__init__(destination, sample_rate, dtype, container_format)
        self.container = av.open(self.destination, "w", format=self.container_format)
        self.stream = None

    def open(self):
        kwargs = {"layout": standard_channel_layouts[self.num_channels][0]}
        audio_codec, audio_format = self.guess_codec_format()
        if audio_format is not None:
            kwargs["format"] = audio_format
        self.stream = self.container.add_stream(audio_codec, self.sample_rate, **kwargs)

    def guess_codec_format(self) -> tuple[str, str | None]:
        default_codec = self.container.default_audio_codec
        if default_codec is None:
            raise ValueError(f"Container {self.container.format.name} has no default audio codec")
        if self.dtype is None:
            return default_codec, None
        dtype_format = dtype_formats.get(self.dtype)
        if dtype_format is None:
            raise ValueError(f"Unsupported output dtype: {self.dtype.name}")
        for audio_format in av.Codec(default_codec, "w").audio_formats or []:
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
        raise ValueError(f"No {self.container.format.name} audio encoder supports dtype {self.dtype.name}")

    def write(self, audio: np.ndarray):
        audio = self.prepare_audio(audio)
        if self.stream is None:
            self.open()
        audio_frame = from_ndarray(audio, self.stream.format.name, self.stream.layout, self.stream.rate)
        for packet in self.stream.encode(audio_frame):
            self.container.mux(packet)

    def close(self):
        container = self.container
        stream = self.stream
        self.container = None
        self.stream = None
        failure = None
        try:
            if container is not None and stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as error:
            failure = error
        try:
            if container is not None:
                container.close()
        except Exception as error:
            if failure is None:
                failure = error
        finally:
            super().close()
        if failure is not None:
            raise failure
