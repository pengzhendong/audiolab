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

import numpy as np
from bv.audio.frame import format_dtypes
from lhotse import Seconds
from numpy.random import randint, uniform

from audiolab.av.format import AudioFormat, get_format_dtype
from audiolab.av.frame import clip, from_ndarray, split_audio_frame, to_ndarray
from audiolab.av.layout import AudioLayout


class TestFrame:

    @staticmethod
    def generate_ndarray(nb_channels: int, samples: int, dtype: np.dtype) -> np.ndarray:
        if np.dtype(dtype).kind in ("i", "u"):
            min_value = np.iinfo(dtype).min
            max_value = np.iinfo(dtype).max
            return randint(min_value, max_value, size=(nb_channels, samples), dtype=dtype)
        else:
            return uniform(-1, 1, size=(nb_channels, samples)).astype(dtype)

    def test_clip(self):
        dtypes = (np.uint8, np.int16, np.int32, np.float32, np.float64)
        for source_dtype in dtypes:
            for target_dtype in dtypes:
                target_dtype = np.dtype(target_dtype)
                ndarray = self.generate_ndarray(1, 42, source_dtype)
                ndarray = clip(ndarray, target_dtype)
                if target_dtype.kind in ("i", "u"):
                    min_value = np.iinfo(target_dtype).min
                    max_value = np.iinfo(target_dtype).max
                    assert np.all(ndarray >= min_value) and np.all(ndarray <= max_value)
                else:
                    assert np.all(ndarray >= -1) and np.all(ndarray <= 1)

    def test_from_to_ndarray(self):
        for layout_name in ("mono", "stereo", "2.1", "3.0"):
            layout = AudioLayout[layout_name].value
            nb_channels = layout.nb_channels
            for format_name in format_dtypes.keys():
                format = AudioFormat[format_name].value
                dtype = get_format_dtype(format)
                for rate in (8000, 16000, 24000, 48000):
                    ndarray = self.generate_ndarray(nb_channels, rate, dtype)
                    frame = from_ndarray(ndarray, format, layout, rate)
                    assert frame.format.name == format.name
                    assert frame.layout.name == layout.name
                    assert frame.rate == rate
                    assert np.allclose(to_ndarray(frame), ndarray)

    def test_split_audio_frame(self):
        pts = 0
        for layout_name in ("mono", "stereo", "2.1", "3.0"):
            layout = AudioLayout[layout_name].value
            nb_channels = layout.nb_channels
            for format_name in format_dtypes.keys():
                format = AudioFormat[format_name].value
                dtype = get_format_dtype(format)
                for rate in (8000, 16000, 24000, 48000):
                    duration: Seconds = randint(0, 10)
                    offset: Seconds = randint(0, 10)
                    duration_samples = int(duration * rate)
                    offset_samples = int(min(offset, duration) * rate)
                    ndarray = self.generate_ndarray(nb_channels, duration_samples, dtype)
                    frame = from_ndarray(ndarray, format, layout, rate, pts=pts)
                    left, right = split_audio_frame(frame, offset)
                    if offset > 0:
                        assert left.rate == rate
                        assert left.format.name == format.name
                        assert left.layout.name == layout.name
                        assert left.pts == pts
                        assert left.samples == offset_samples
                    else:
                        assert left is None
                    if offset <= duration:
                        assert right.rate == rate
                        assert right.format.name == format.name
                        assert right.layout.name == layout.name
                        assert right.pts == pts + offset_samples
                        assert right.samples == duration_samples - offset_samples
                    else:
                        assert right is None
