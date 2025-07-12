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

from typing import Any

import click

import audiolab


@click.command()
@click.argument("audio-file", type=click.File(mode="rb"), default="-")
@click.option("--stream-id", "-s", type=int, default=0)
@click.option("--force-decoding", "-f", is_flag=True)
def info(audio_file: Any, stream_id: int = 0, force_decoding: bool = False):
    """
    Print the information of an audio file.

    Args:
        audio_file: The audio file, path to audio file, or stdin.
        stream_id: The index of the stream to load.
        force_decoding: Whether to force decoding the audio file to get the duration.
    """
    print(audiolab.info(audio_file, stream_id, force_decoding))
