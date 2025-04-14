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

from typing import Union

import av
import numpy as np

from audiolab.pyav.utils import format_dtypes, get_template


class AudioFormat:
    def __init__(self, name: Union[str, av.AudioFormat]):
        if isinstance(name, str):
            self.name = name
            self.format = av.AudioFormat(name)
        else:
            self.name = name.name
            self.format = name
        self.codecs = set()
        self.bits = self.format.bits
        self.bytes = self.format.bytes
        self.dtype = np.dtype(format_dtypes[self.name])

    @property
    def __doc__(self):
        return get_template("format").render(format=self.format, dtype=self.dtype, codecs=self.codecs)

    def __repr__(self):
        codecs = [codec.name for codec in self.codecs]
        return f"{self.format} ({', '.join(codecs)})"
