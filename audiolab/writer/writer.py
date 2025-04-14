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

from fractions import Fraction
from typing import Any, Dict, Optional, Union

import av
import numpy as np
from av import AudioFrame, AudioLayout

from audiolab.pyav import from_ndarray


class Writer:
    def __init__(
        self,
        file: Any,
        codec: Union[str, av.Codec],
        rate: Optional[Union[int, Fraction]] = None,
        layout: Optional[Union[int, str, AudioLayout]] = None,
        options: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.container = av.open(file, "w")
        if isinstance(codec, av.Codec):
            codec = codec.name
        if isinstance(layout, int):
            assert layout in (1, 2)
            layout = "mono" if layout == 1 else "stereo"
        kwargs["layout"] = layout
        self.stream = self.container.add_stream(codec, rate, options, **kwargs)

    def write(self, frame: Union[AudioFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            frame = from_ndarray(frame, self.stream.format.name, self.stream.layout, self.stream.rate)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        self.container.close()
