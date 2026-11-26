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

"""Fixtures shared by the CLI utils tests."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _is_loaded_from(module: ModuleType, roots: Sequence[Path]) -> bool:
  """Returns whether `module` was loaded from a file or dir under `roots`."""
  locations = list(getattr(module, "__path__", None) or [])
  if module_file := getattr(module, "__file__", None):
    locations.append(module_file)
  return any(
      Path(location).resolve().is_relative_to(root)
      for location in locations
      for root in roots
  )


@pytest.fixture(autouse=True)
def isolate_agent_imports() -> Iterator[None]:
  """Restores sys.path and drops the agent modules each test imports."""
  original_path = sys.path.copy()
  original_modules = set(sys.modules)
  yield
  # Agents load from the agents_dir AgentLoader adds to sys.path.
  added_roots = [
      Path(entry).resolve() for entry in sys.path if entry not in original_path
  ]
  # Drop this test's agents from sys.modules so later tests can reuse names.
  # Collect first: a namespace package's __path__ reads its parent's entry.
  stale_names = [
      name
      for name in set(sys.modules) - original_modules
      if _is_loaded_from(sys.modules[name], added_roots)
  ]
  for name in stale_names:
    del sys.modules[name]
  sys.path[:] = original_path
