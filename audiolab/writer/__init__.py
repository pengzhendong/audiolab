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

import numpy as np
from av import AudioFrame

from .writer import Writer


def save_audio(
    file: Any,
    frame: Union[AudioFrame, np.ndarray],
    rate: Optional[Union[int, Fraction]] = None,
    options: Optional[Dict[str, str]] = None,
    **kwargs
):
    if isinstance(frame, np.ndarray):
        assert frame.dtype in [np.int16, np.float32]
        assert frame.ndim == 2 and frame.shape[0] in [1, 2]
        codec_name = "pcm_s16le" if frame.dtype == np.int16 else "pcm_f32le"
        format = "s16" if frame.dtype == np.int16 else "flt"
        layout = "mono" if frame.shape[0] == 1 else "stereo"
    else:
        assert frame.format.name in ["s16", "flt"]
        assert frame.layout.name in ["mono", "stereo"]
        codec_name = "pcm_s16le" if frame.format.name == "s16" else "pcm_f32le"
        format = frame.format.name
        layout = frame.layout
    writer = Writer(file, codec_name, rate, options, format=format, layout=layout, **kwargs)
    writer.write(frame)
    writer.close()


__all__ = ["Writer", "save_audio"]
