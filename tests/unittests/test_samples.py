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

import json
from pathlib import Path

from google.adk.apps.app import App
from google.adk.cli.agent_test_runner import test_agent_replay as _test_agent_replay
from google.genai import types
import pytest

CONTRIBUTING_DIR = Path(__file__).parent.parent.parent / "contributing"


def get_test_files():
  """Yields (sample_dir, test_file_path)."""
  if not CONTRIBUTING_DIR.exists():
    return
  for test_file in CONTRIBUTING_DIR.rglob("tests/*.json"):
    sample_dir = test_file.parent.parent
    if (
        (sample_dir / "agent.py").exists()
        or (sample_dir / "__init__.py").exists()
        or (sample_dir / "root_agent.yaml").exists()
    ):
      try:
        rel_dir = sample_dir.relative_to(CONTRIBUTING_DIR)
        test_id = f"{rel_dir}/{test_file.name}"
      except ValueError:
        test_id = f"{sample_dir.name}/{test_file.name}"

      if test_file.stem.endswith("_xfail"):
        yield pytest.param(
            sample_dir, test_file, id=test_id, marks=pytest.mark.xfail
        )
      else:
        yield pytest.param(sample_dir, test_file, id=test_id)


@pytest.mark.parametrize(
    "sample_dir, test_file",
    list(get_test_files()),
)
def test_sample(sample_dir: Path, test_file: Path, monkeypatch):
  """Tests a sample by replaying exported session events."""
  _test_agent_replay(sample_dir, test_file, monkeypatch)
