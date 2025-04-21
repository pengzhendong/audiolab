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

from importlib.resources import files
from io import BytesIO

from jinja2 import Environment, FileSystemLoader
from lhotse.caching import AudioCache
from lhotse.utils import SmartOpen

loader = FileSystemLoader(files("audiolab.av").joinpath("templates"))


def get_template(name: str) -> str:
    return Environment(loader=loader).get_template(f"{name}.txt")


def load_url(url: str) -> BytesIO:
    audio_bytes = AudioCache.try_cache(url)
    if audio_bytes is None:
        with SmartOpen.open(url, "rb") as f:
            audio_bytes = f.read()
        AudioCache.add_to_cache(url, audio_bytes)
    return BytesIO(audio_bytes)
