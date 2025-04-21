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

from typing import Dict

import bv

from audiolab.av.typing import AudioLayoutEnum

"""
$ ffmpeg -layouts
"""
standard_channel_layouts = (
    "mono",
    "stereo",
    "2.1",
    "3.0",
    "3.0(back)",
    "4.0",
    "quad",
    "quad(side)",
    "3.1",
    "5.0",
    "5.0(side)",
    "4.1",
    "5.1",
    "5.1(side)",
    "6.0",
    "6.0(front)",
    "3.1.2",
    "hexagonal",
    "6.1",
    "6.1(back)",
    "6.1(front)",
    "7.0",
    "7.0(front)",
    "7.1",
    "7.1(wide)",
    "7.1(wide-side)",
    "5.1.2",
    "octagonal",
    "cube",
    "5.1.4",
    "7.1.2",
    "7.1.4",
    "7.2.3",
    "9.1.4",
    "hexadecagonal",
    "downmix",
    "22.2",
)
audio_layouts: Dict[str, bv.AudioLayout] = {name: bv.AudioLayout(name) for name in standard_channel_layouts}
AudioLayout = AudioLayoutEnum("AudioLayout", audio_layouts)
