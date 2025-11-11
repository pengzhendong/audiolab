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
from typing import Any, Iterator, List, Optional

import numpy as np

from audiolab.av import aformat, load_url, split_audio_frame
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFormat, AudioFrame, Dtype, Filter, Seconds
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
        cache_url: bool = False,
        **kwargs,
    ):
        """
        Create a Reader object.

        Args:
            file: The audio file, audio url, path to audio file, bytes of audio data, etc.
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
            cache_url: Whether to cache the audio file.
            always_2d: Whether to return 2d ndarrays even if the audio frame is mono.
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
            frame_size = int(frame_size_ms * self.stream.rate // 1000)
        elif frame_size is None:
            frame_size = np.iinfo(np.uint32).max
        self.frame_size = min(frame_size, np.iinfo(np.uint32).max)
        self.graph = Graph(
            self.stream, filters=filters, frame_size=frame_size, **kwargs
        )
        self.always_2d = kwargs.get("always_2d", True)

    @property
    def num_frames(self) -> int:
        """
        Get the number of the audio frames in the audio stream.
        Note: Filters may change the number of frames.
        """
        return math.ceil(self.duration * self.rate / self.frame_size)

    def __iter__(self) -> Iterator[AudioFrame]:
        for frame in self.container.decode(self.stream):
            assert frame.time == float(frame.pts * self.stream.time_base)
            if frame.time > self.end_time:
                break
            if frame.time + frame.samples / frame.rate > self.end_time:
                frame, _ = split_audio_frame(frame, self.end_time - frame.time)
            self.graph.push(frame)
            yield from self.graph.pull()
        yield from self.graph.pull(partial=True)

    def load_audio(self, always_2d: Optional[bool] = None) -> AudioFrame:
        """
        Load the audio stream into a numpy array.

        Returns:
            The numpy array of the audio stream.
        """
        if always_2d is None:
            always_2d = self.always_2d
        frames, rates = zip(*self)
        assert len(set(rates)) == 1
        return np.concatenate(frames, axis=1 if always_2d else 0), rates[0]
