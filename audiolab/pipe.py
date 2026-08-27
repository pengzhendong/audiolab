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

from collections.abc import Iterator

import numpy as np
from numpy.typing import DTypeLike

from audiolab.av import Graph, build_filter_chain
from audiolab.av.frame import pad
from audiolab.av.typing import AudioFormatLike, FilterSpec


class AudioPipe:
    def __init__(
        self,
        input_sample_rate: int,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        is_planar: bool = False,
        sample_format: AudioFormatLike | None = None,
        output_sample_rate: int | None = None,
        to_mono: bool = False,
        frame_size: int | None = 1024,
        fill_value: float | None = None,
        always_2d: bool = True,
    ):
        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive")
        if output_sample_rate is not None and output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        if fill_value is not None and frame_size is None:
            raise ValueError("frame_size is required when fill_value is set")
        self.input_sample_rate = input_sample_rate
        self.graph = None
        self.filters = build_filter_chain(
            filters,
            dtype=dtype,
            is_planar=is_planar,
            sample_format=sample_format,
            sample_rate=output_sample_rate,
            to_mono=to_mono,
        )
        self.frame_size = frame_size
        self.fill_value = fill_value
        self.always_2d = always_2d

    def push(self, audio: np.ndarray) -> None:
        audio = np.atleast_2d(audio)
        if self.graph is None:
            self.graph = Graph(
                sample_rate=self.input_sample_rate,
                dtype=audio.dtype,
                channels=audio.shape[0],
                filters=self.filters,
                frame_size=self.frame_size,
            )
        self.graph.push(audio)

    def pull(self, partial: bool = False) -> Iterator[tuple[np.ndarray, int]]:
        if self.graph is None:
            return
        for audio, sample_rate in self.graph.pull(partial=partial):
            if self.fill_value is not None:
                audio = pad(audio, self.frame_size, self.fill_value)
            yield audio if self.always_2d else audio.squeeze(), sample_rate
