# Copyright 2026 Google LLC
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

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

# verify_md and run are standalone scripts (not a package), so load them by path.
# The scripts dir lives at the repo root on GitHub but under
# "open_source_workspace/" in the source tree, so try both layouts.
_ADK_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_SUBPATH = (
    Path(".agents") / "skills" / "adk-verify-snippets" / "scripts"
)
SCRIPTS_DIR = next(
    (
        candidate
        for candidate in (
            _ADK_ROOT / _SCRIPTS_SUBPATH,
            _ADK_ROOT / "open_source_workspace" / _SCRIPTS_SUBPATH,
        )
        if candidate.exists()
    ),
    _ADK_ROOT / _SCRIPTS_SUBPATH,
)


def import_script(name: str) -> ModuleType:
  file_path = SCRIPTS_DIR / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script {name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


verify_md = import_script("verify_md")
run_module = import_script("run")


def test_clean_name() -> None:
  assert verify_md.clean_name("Hello World 123!") == "hello_world_123"
  assert (
      verify_md.clean_name("Verify-Snippets: Ignore") == "verifysnippets_ignore"
  )


def test_md_cell() -> None:
  assert verify_md.md_cell("col1 | col2") == r"col1 \| col2"


def test_safe_fence() -> None:
  assert verify_md.safe_fence("x = 1", "python") == "```python\nx = 1\n```"
  assert (
      verify_md.safe_fence("x = ```foo```", "python")
      == "````python\nx = ```foo```\n````"
  )


def test_extract_snippets(tmp_path: Path) -> None:
  md_content = """
# Heading 1
Some description here.

```python
import os
print(os.getcwd())
```

## Heading 2

<!-- verify-snippets: ignore -->
```python
# ignored snippet
x = 1 + 2
```
"""
  md_file = tmp_path / "test.md"
  md_file.write_text(md_content, encoding="utf-8")

  snippets = verify_md.extract_snippets(md_file)
  assert len(snippets) == 2

  assert snippets[0]["heading"] == "Heading 1"
  assert "import os" in snippets[0]["code"]
  assert snippets[0]["skip"] is False

  assert snippets[1]["heading"] == "Heading 2"
  assert "# ignored snippet" in snippets[1]["code"]
  assert snippets[1]["skip"] is True


def test_extract_error_detail() -> None:
  stdout = "Some progress...\n❌ Run Failure: error occurred\n"
  stderr = (
      'Traceback (most recent call last):\n  File "run.py", line'
      " 12\nValueError: invalid value\n"
  )
  detail = verify_md.extract_error_detail(stdout, stderr)
  assert detail == "`ValueError: invalid value`"

  # Fallback to stdout
  stdout_err = "ValueError: error in stdout\n"
  detail_stdout = verify_md.extract_error_detail(stdout_err, "")
  assert detail_stdout == "`ValueError: error in stdout`"


def test_discover_adk_component() -> None:
  # Create a dummy module to test discovery
  class DummyModule:
    pass

  dummy = DummyModule()

  # When there is nothing, should return (None, None)
  component, comp_type = run_module.discover_adk_component(dummy)
  assert component is None
  assert comp_type is None
