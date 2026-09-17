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

"""Tests for the MCP SDK dependency seam."""

from __future__ import annotations

import ast
import os
import pathlib

from google.adk.dependencies import _mcp as mcp_dependency
from google.adk.dependencies import _mcp_name

_ADK_ROOT = pathlib.Path(mcp_dependency.__file__).resolve().parent.parent

# Not part of the released package, so these may name the internal SDK copy
# directly.
_NOT_SHIPPED = (
    'tests/',
    'internal/',
    'platform/internal/',
    'dependencies_internal/',
    # Real files, but sample code rather than library code, and not part of the
    # importable package.
    'open_source_workspace/',
)


def _exported_library_sources() -> list[pathlib.Path]:
  """Every ADK source file that ships and is not a test."""
  found = []
  for dirpath, dirnames, filenames in os.walk(_ADK_ROOT):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for filename in filenames:
      if not filename.endswith('.py'):
        continue
      path = pathlib.Path(dirpath) / filename
      relative = path.relative_to(_ADK_ROOT).as_posix()
      if relative.startswith(_NOT_SHIPPED):
        continue
      # The seam itself is the one place allowed to name the SDK. It is reached
      # as `dependencies/_mcp.py`, but resolving the symlink an open-source
      # build uses lands on the flavor's real name.
      if relative in ('dependencies/_mcp.py', 'dependencies_external/_mcp.py'):
        continue
      found.append(path)
  return found


def _sdk_imports(source: str) -> list[str]:
  """Lines in `source` that import the MCP SDK by its own name."""
  tree = ast.parse(source)
  lines = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
      if node.module and (
          node.module == 'mcp' or node.module.startswith('mcp.')
      ):
        lines.append(f'{node.lineno}: from {node.module} import ...')
    elif isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name == 'mcp' or alias.name.startswith('mcp.'):
          lines.append(f'{node.lineno}: import {alias.name}')
  return lines


class TestTheSeamHolds:
  """The seam is only worth having if nothing routes around it."""

  def test_no_shipped_module_imports_the_sdk_directly(self):
    """This is the regression pin for the whole arrangement.

    A single `from mcp import ...` added back to a shipped module is invisible
    until an export produces a package that names an SDK the released wheel
    cannot install. Catch it here instead.
    """
    offenders = {}
    for path in _exported_library_sources():
      found = _sdk_imports(path.read_text(encoding='utf-8'))
      if found:
        offenders[path.relative_to(_ADK_ROOT).as_posix()] = found

    assert not offenders, (
        'These shipped modules import the MCP SDK directly. Import from'
        f' `google.adk.dependencies._mcp` instead: {offenders}'
    )

  def test_every_advertised_name_resolves(self):
    """`__all__` is the contract callers rely on, so it must be honest."""
    missing = [
        name
        for name in mcp_dependency.__all__
        if getattr(mcp_dependency, name, None) is None
    ]

    assert not missing

  def test_the_advertised_sdk_name_is_the_one_the_seam_imports(self):
    """Telemetry looks the SDK up in `sys.modules` by this name.

    `_mcp_name` spells the name a second time so that callers can read it
    without paying for the import. A rename that misses one of the two files
    leaves that lookup silently false.
    """
    imported = mcp_dependency.ClientSession.__module__.partition('.')[0]

    assert imported == _mcp_name.SDK_MODULE_NAME
