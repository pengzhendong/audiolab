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


from typing import Literal, Union

import av

from audiolab.pyav.utils import get_template


class AudioCodec:
    def __init__(self, name: Union[str, av.Codec], mode: Literal["r", "w"] = "r"):
        if isinstance(name, str):
            self.name = name
            self.codec = av.Codec(name, mode)
        else:
            self.name = name.name
            self.codec = name
        self.formats = set()
        self.long_name = self.codec.long_name

    @property
    def __doc__(self):
        return get_template("codec").render(codec=self.codec, formats=self.formats)

    def __repr__(self):
        formats = [format.name for format in self.formats]
        return f"{self.codec} ({', '.join(formats)})"
