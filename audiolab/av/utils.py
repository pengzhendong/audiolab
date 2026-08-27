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

import logging
import sys
from functools import lru_cache

import numpy as np
from jinja2 import Environment, PackageLoader, Template
from numpy.random import randint, uniform

loader = PackageLoader("audiolab.av", "templates")
environment = Environment(loader=loader)


def generate_ndarray(num_channels: int, samples: int, dtype: np.dtype, always_2d: bool = True) -> np.ndarray:
    if np.dtype(dtype).kind in ("i", "u"):
        audio = randint(np.iinfo(dtype).min, np.iinfo(dtype).max, size=(num_channels, samples), dtype=dtype)
    else:
        audio = uniform(-1, 1, size=(num_channels, samples)).astype(dtype)
    return audio if always_2d else audio.squeeze()


def get_template(name: str) -> Template:
    return environment.get_template(f"{name}.txt")


@lru_cache(maxsize=128)
def get_logger(name, level=logging.INFO):
    logger = logging.Logger(name, level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
