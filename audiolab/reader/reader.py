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

import math
from typing import Any, List, Optional

import numpy as np

from audiolab.av import AudioGraph, aformat, load_url, split_audio_frame
from audiolab.av.typing import AudioFormat, Dtype, Filter, Seconds
from audiolab.reader.info import Info


class Reader(Info):
    def __init__(
        self,
        file: Any,
        stream_id: int = 0,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        filters: Optional[List[Filter]] = None,
        dtype: Optional[Dtype] = None,
        is_planar: bool = False,
        format: Optional[AudioFormat] = None,
        rate: Optional[int] = None,
        to_mono: bool = False,
        frame_size: Optional[int] = None,
        frame_size_ms: Optional[int] = None,
        return_ndarray: bool = True,
        cache_url: bool = True,
    ):
        """
        Create a Reader object.

        Args:
            file: The audio file, path to audio file, bytes of audio data, etc.
            stream_id: The index of the stream to load.
            offset: The offset of the audio stream to load.
            duration: The duration of the audio stream to load.
            filters: The filters to apply to the audio stream.
            dtype: The data type of the audio frames.
            is_planar: Whether the audio frames are planar.
            format: The format of the audio frames.
            rate: The sample rate of the audio frames.
            to_mono: Whether to convert the audio frames to mono.
            frame_size: The frame size of the audio frames.
            frame_size_ms: The frame size in milliseconds of the audio frames.
            return_ndarray: Whether to return the audio frames as ndarrays.
            cache_url: Whether to cache the audio file.
        Returns:
            The Reader object.
        """
        if isinstance(file, str) and "://" in file and cache_url:
            file = load_url(file, cache=True)

        super().__init__(file, stream_id)
        self.start_time = int(offset / self.stream.time_base)
        self.end_time = Seconds("inf") if duration is None else offset + duration
        if self.start_time > 0:
            self.container.seek(self.start_time, any_frame=True, stream=self.stream)

        if not all([dtype is None, format is None, rate is None, not to_mono]):
            filters = filters or []
            filters.append(aformat(dtype, is_planar, format, rate, to_mono))

        if frame_size_ms is not None:
            frame_size = frame_size_ms * self.stream.rate / 1000
        else:
            frame_size = frame_size or np.iinfo(np.uint32).max
        self.frame_size = min(frame_size, np.iinfo(np.uint32).max)
        self.graph = AudioGraph(
            stream=self.stream, filters=filters, frame_size=frame_size, return_ndarray=return_ndarray
        )

    @property
    def num_frames(self) -> int:
        """
        Get the number of the input audio frames in the audio stream.

        Returns:
            The number of the input audio frames in the audio stream.
        """
        return math.ceil(self.duration * self.rate / self.frame_size)

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            assert frame.time == float(frame.pts * self.stream.time_base)
            if frame.time > self.end_time:
                break
            if frame.time + frame.samples / frame.rate > self.end_time:
                frame, _ = split_audio_frame(frame, self.end_time - frame.time)
            self.graph.push(frame)
            yield from self.graph.pull()
        yield from self.graph.pull(partial=True)
