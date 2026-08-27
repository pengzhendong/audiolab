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
from click.testing import CliRunner

from audiolab.cli import main
from audiolab.writer import save_audio


def test_cli_reads_audio_with_registered_backend_name(tmp_path):
    audio_path = tmp_path / "audio.wav"
    save_audio(audio_path, np.zeros((1, 800), dtype=np.int16), sample_rate=16_000)

    result = CliRunner().invoke(main, [str(audio_path)])

    assert result.exit_code == 0, result.output
    assert "Sample Rate    : 16000" in result.output
