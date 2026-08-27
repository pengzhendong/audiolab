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

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = [*sorted(ROOT.glob("*.md")), *sorted((ROOT / "docs").glob("*.md"))]


def test_python_documentation_examples_are_valid_syntax():
    for path in MARKDOWN_FILES:
        contents = path.read_text()
        for index, example in enumerate(re.findall(r"```python\n(.*?)```", contents, re.DOTALL), start=1):
            ast.parse(example, filename=f"{path.name}:python-block-{index}")


def test_local_documentation_links_exist():
    for path in MARKDOWN_FILES:
        contents = path.read_text()
        for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", contents):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            assert destination.exists(), f"Broken link in {path.relative_to(ROOT)}: {target}"
