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

import av

from audiolab.av.typing import AudioLayoutEnum

"""
$ ffmpeg -layouts
"""
standard_channel_layouts = {
    0: ["downmix"],
    1: ["mono"],
    2: ["stereo"],
    3: ["2.1", "3.0", "3.0(back)"],
    4: ["4.0", "quad", "quad(side)", "3.1"],
    5: ["5.0", "5.0(side)", "4.1"],
    6: ["5.1", "6.0", "6.0(front)", "hexagonal", "5.1(side)", "3.1.2"],
    7: ["7.0", "7.0(front)", "6.1", "6.1(back)", "6.1(front)"],
    8: ["7.1", "7.1(wide)", "7.1(wide-side)", "cube", "octagonal", "5.1.2"],
    10: ["5.1.4", "7.1.2"],
    12: ["7.1.4", "7.2.3"],
    14: ["9.1.4"],
    16: ["hexadecagonal"],
    24: ["22.2"],
}

audio_layouts: Dict[str, av.AudioLayout] = {
    name: av.AudioLayout(name) for layouts in standard_channel_layouts.values() for name in layouts
}
AudioLayout = AudioLayoutEnum("AudioLayout", audio_layouts)
