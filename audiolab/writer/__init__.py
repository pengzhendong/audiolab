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

from typing import Any, Optional

import av

from audiolab.av.format import get_dtype
from audiolab.av.frame import to_ndarray
from audiolab.av.typing import AudioFrame
from audiolab.writer.writer import Writer


def save_audio(file: Any, frame: AudioFrame, rate: Optional[int] = None, **kwargs):
    _rate = None
    if isinstance(frame, tuple):
        frame, _rate = frame
    if isinstance(frame, av.AudioFrame):
        if kwargs.get("format", None) is None:
            kwargs["dtype"] = kwargs.get("dtype", get_dtype(frame.format))
        _rate = frame.rate
        frame = to_ndarray(frame)
    if rate is None:
        assert _rate is not None
        rate = _rate
    elif _rate is not None:
        assert rate == _rate

    writer = Writer(file, rate, **kwargs)
    writer.write(frame)
    writer.close()


__all__ = ["Writer", "save_audio"]
