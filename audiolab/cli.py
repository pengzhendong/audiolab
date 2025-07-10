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

from io import BytesIO
from typing import Any

import click

import audiolab


@click.command()
@click.argument("audio-file", type=click.File(mode="rb"), default="-")
@click.option("--stream-id", type=int, default=0)
@click.option("--force-duration", "-f", is_flag=True)
def info(audio_file: Any, stream_id: int = 0, force_duration: bool = False):
    if audio_file.name == "-":
        audio_file = BytesIO(audio_file.read())
    print(audiolab.info(audio_file, stream_id, force_duration))
