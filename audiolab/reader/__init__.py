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

import numpy as np
from soundfile import LibsndfileError

from audiolab.av import aformat
from audiolab.av.graph import Graph
from audiolab.reader.info import Info
from audiolab.reader.reader import Reader
from audiolab.reader.stream_reader import StreamReader


def info(source: Any, forced_decoding: bool = False, backends: list[str] | None = None) -> Info:
    """
    Get the information of an audio file.

    Args:
        source: The input audio file, audio URL, path, or encoded bytes.
        forced_decoding: Whether to forced decoding the audio file to get the duration.
        backends: The list of backends to use to get the information.
    Returns:
        The information of the audio file.
    """
    return Info(source, forced_decoding=forced_decoding, backends=backends)


def load_audio(source: Any, **kwargs) -> tuple[np.ndarray, int]:
    """Decode an entire audio source into memory.

    ``Reader`` is the streaming API. This convenience function is deliberately
    eager and always returns one audio array plus its sample rate, regardless
    of the reader's internal ``frame_size``.
    """
    chunks = []
    output_rate = None
    with Reader(source, **kwargs) as reader:
        try:
            for chunk, chunk_rate in reader:
                if output_rate is not None and output_rate != chunk_rate:
                    raise RuntimeError("Audio sample rate changed while decoding")
                output_rate = chunk_rate
                chunks.append(chunk)
        except LibsndfileError as error:
            if str(error) != "Internal psf_fseek() failed.":
                raise

        if not chunks:
            always_2d = kwargs.get("always_2d", True)
            shape = (0, 0) if always_2d else (0,)
            return np.empty(shape, dtype=reader.dtype), kwargs.get("sample_rate") or reader.sample_rate

    axis = 1 if chunks[0].ndim == 2 else 0
    if output_rate is None:
        raise RuntimeError("Decoded audio did not provide a sample rate")
    return np.concatenate(chunks, axis=axis), output_rate


__all__ = ["Graph", "Reader", "StreamReader", "aformat", "load_audio"]
