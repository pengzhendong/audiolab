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
from typing import Iterator, List, Optional

import av
import numpy as np

from audiolab.av import AudioGraph, aformat
from audiolab.av.typing import AudioFormat, AudioFrame, Dtype, Filter
from audiolab.reader import info
from audiolab.writer import save_audio


def stream_template(frame: AudioFrame, rate: Optional[int] = None) -> av.AudioStream:
    """
    Get a stream template of the audio frame.

    Args:
        frame: The audio frame.
        rate: The sample rate of the audio frame.
    Returns:
        The stream template.
    """
    rate = rate or (frame[1] if isinstance(frame, tuple) else frame.rate)
    bytes_io = BytesIO()
    save_audio(bytes_io, frame, rate=rate)
    return info(bytes_io).stream


class AudioPipe:
    def __init__(
        self,
        in_rate: int,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        out_rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = 1024,
        return_ndarray: bool = True,
        always_2d: bool = True,
    ):
        """
        Create a Pipe object.

        Args:
            in_rate: The sample rate of the input audio frames.
            filters: The filters to apply to the audio pipe.
            dtype: The data type of the output audio frames.
            is_planar: Whether the output audio frames are planar.
            format: The format of the output audio frames.
            out_rate: The sample rate of the output audio frames.
            to_mono: Whether to convert the output audio frames to mono.
            frame_size: The frame size of the audio frames.
            return_ndarray: Whether to return the audio frames as ndarrays.
        """
        self.in_rate = in_rate
        self.graph = None
        if not all([dtype is None, format is None, out_rate is None, to_mono is None]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, out_rate, to_mono))
        self.filters = filters
        self.frame_size = frame_size
        self.return_ndarray = return_ndarray
        self.always_2d = always_2d

    def push(self, frame: np.ndarray):
        """
        Push a frame of audio data to the audio pipe.

        Args:
            frame: The frame of audio data to push.
        """
        if self.graph is None:
            self.graph = AudioGraph(
                stream=stream_template(frame, self.in_rate),
                filters=self.filters,
                frame_size=self.frame_size,
                return_ndarray=self.return_ndarray,
                always_2d=self.always_2d,
            )
        self.graph.push(frame)

    def pull(self, partial: bool = False) -> Iterator[AudioFrame]:
        """
        Pull an audio frame from the audio pipe.

        Args:
            partial: Whether to pull a partial frame.
        Yields:
            The audio frame.
        """
        yield from self.graph.pull(partial=partial)
