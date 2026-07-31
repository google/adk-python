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

"""Runs snippets in a fresh interpreter to assert import-time side effects.

Imports are process-global and irreversible, so any assertion about what a
package pulls in has to happen in a process that has not already imported it.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / 'src'


def run_isolated(source: str) -> subprocess.CompletedProcess[str]:
  """Runs source against this checkout in a fresh Python process."""
  env = os.environ.copy()
  source_path = str(SOURCE_ROOT)
  current_pythonpath = env.get('PYTHONPATH')
  env['PYTHONPATH'] = (
      source_path
      if not current_pythonpath
      else os.pathsep.join((source_path, current_pythonpath))
  )
  # Run from an empty directory so the interpreter's implicit sys.path[0] entry
  # cannot shadow a stdlib module: ADK ships a ``platform`` package that would
  # otherwise mask stdlib ``platform`` (breaking uuid, pydantic, ...) here.
  with tempfile.TemporaryDirectory() as isolated_cwd:
    return subprocess.run(
        [sys.executable, '-c', source],
        cwd=isolated_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def assert_modules_unloaded(source: str, forbidden: tuple[str, ...]) -> None:
  """Asserts source leaves every forbidden module (and submodule) unimported."""
  result = run_isolated(f"""
import sys
{source}

forbidden = {forbidden!r}
loaded = [
    prefix
    for prefix in forbidden
    if any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    )
]
assert not loaded, loaded
""")
  assert result.returncode == 0, result.stderr
