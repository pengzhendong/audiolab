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
from importlib.resources import files

import numpy as np
from jinja2 import Environment, FileSystemLoader
from numpy.random import randint, uniform

loader = FileSystemLoader(files("audiolab.av").joinpath("templates"))


def generate_ndarray(nb_channels: int, samples: int, dtype: np.dtype, always_2d: bool = True) -> np.ndarray:
    if np.dtype(dtype).kind in ("i", "u"):
        ndarray = randint(np.iinfo(dtype).min, np.iinfo(dtype).max, size=(nb_channels, samples), dtype=dtype)
    else:
        ndarray = uniform(-1, 1, size=(nb_channels, samples)).astype(dtype)
    return ndarray if always_2d else ndarray.squeeze()


def get_template(name: str) -> str:
    return Environment(loader=loader).get_template(f"{name}.txt")


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.propagate = False
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
