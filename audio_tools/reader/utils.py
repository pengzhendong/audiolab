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

from typing import Any, Dict, List, Optional, Tuple, Union

import av
import numpy as np
from av import AudioFrame
from av.filter import Graph


def build_graph(stream: av.AudioStream, filters: Optional[List[Tuple[str, str]]] = None) -> Graph:
    graph = Graph()
    graph.link_nodes(
        graph.add_abuffer(template=stream),
        *[graph.add(filter[0], filter[1]) for filter in filters] if filters else [],
        graph.add("abuffersink"),
    ).configure()
    return graph


def db_to_linear(db):
    """
    Convert decibels (dB) to linear amplitude ratio.
    """
    return 10 ** (db / 20)


def parse_frequency(frequency: Union[str, int]) -> int:
    if isinstance(frequency, str):
        if frequency.endswith(("k", "K")):
            frequency = int(frequency[:-1]) * 1000
        else:
            frequency = int(frequency)
    return frequency


def parse_threshold(threshold: Union[str, int]) -> int:
    if isinstance(threshold, str):
        if threshold.endswith("dB"):
            threshold = db_to_linear(float(threshold[:-2]))
        else:
            threshold = float(threshold)
    return threshold


def format_options(options: Union[Dict[str, Any], List[Union[str, Tuple[str, Any]]]]) -> str:
    if isinstance(options, dict):
        return ":".join(f"{k}={v}" for k, v in options.items())
    else:
        return ":".join(opt if isinstance(opt, str) else f"{opt[0]}={opt[1]}" for opt in options)


def to_ndarray(frame: AudioFrame) -> np.ndarray:
    ndarray = frame.to_ndarray()
    if frame.format.is_packed:
        ndarray = ndarray.reshape(-1, frame.layout.nb_channels).T
    return ndarray
