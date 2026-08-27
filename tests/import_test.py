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

import subprocess
import sys


def test_top_level_import_defers_catalogs_and_remote_transports():
    script = """
import sys
import audiolab
deferred = {'audiolab.av.codec', 'audiolab.av.container', 'requests', 'smart_open'}
loaded = deferred.intersection(sys.modules)
if loaded:
    raise SystemExit(f'eager imports: {sorted(loaded)}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
