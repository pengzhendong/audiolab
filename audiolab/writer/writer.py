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
from av import AudioFrame


class Writer:
    def __init__(
        self,
        file: Any,
        codec_name: str,
        rate: Optional[Union[int, Fraction]] = None,
        options: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.container = av.open(file, "w")
        self.stream = self.container.add_stream(codec_name, rate, options, **kwargs)

    def write(self, frame: Union[AudioFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            frame = AudioFrame.from_ndarray(frame, format=self.stream.format.name, layout=self.stream.layout)
            frame.rate = self.stream.rate
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        self.container.close()
