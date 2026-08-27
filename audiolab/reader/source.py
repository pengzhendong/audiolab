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
from typing import Any, Optional

import requests

from audiolab.av.lhotse import load_url
from audiolab.av.typing import Seconds

URL_REQUEST_TIMEOUT = 10


def prepare_source(
    source: Any,
    *,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    cache_url: bool = False,
):
    if isinstance(source, bytes):
        return BytesIO(source)
    if not isinstance(source, str) or "://" not in source:
        return source

    with requests.head(source, allow_redirects=True, timeout=URL_REQUEST_TIMEOUT) as response:
        source = response.url
    if cache_url or (offset == 0 and duration is None):
        return load_url(source, cache=cache_url)
    return source
