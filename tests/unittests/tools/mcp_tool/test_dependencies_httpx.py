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

"""Tests for the HTTP client dependency seam."""

from __future__ import annotations

import ast
import asyncio
import os
import pathlib
import types

from google.adk.dependencies import _httpx as httpx_dependency
from google.adk.dependencies import _mcp as mcp_dependency

_ADK_ROOT = pathlib.Path(httpx_dependency.__file__).resolve().parent.parent

# Only the MCP transport has to follow the SDK's choice of HTTP client. The
# rest of ADK picks its own and is unaffected.
#
# Google's internal build has a second subtree under the same rule,
# `internal/tools/mcp_tool`, which subclasses the session manager. It is not
# part of the open-source tree this walk can see, so it is guarded separately.
_MUST_USE_THE_SEAM = 'tools/mcp_tool/'

# Both flavors, because either one names a major. Reaching for `httpx2` pins
# the subtree to 2.x exactly the way reaching for `httpx` pins it to 1.x.
_HTTP_CLIENT_LIBRARIES = ('httpx', 'httpx2')


def _names_a_library(module: str) -> bool:
  return any(
      module == library or module.startswith(f'{library}.')
      for library in _HTTP_CLIENT_LIBRARIES
  )


def _direct_httpx_imports(source: str) -> list[str]:
  """Lines in `source` that import an HTTP client library by its own name."""
  tree = ast.parse(source)
  lines = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
      if node.module and _names_a_library(node.module):
        lines.append(f'{node.lineno}: from {node.module} import ...')
    elif isinstance(node, ast.Import):
      for alias in node.names:
        if _names_a_library(alias.name):
          lines.append(f'{node.lineno}: import {alias.name}')
  return lines


class TestTheSeamHolds:
  """The seam is only worth having if the MCP transport does not route around it."""

  def test_no_mcp_tool_module_imports_httpx_directly(self):
    """This is the regression pin for the whole arrangement.

    Under MCP 2.x the SDK is built against `httpx2`, and the client ADK
    constructs is handed to it directly. A module that reaches for `httpx`
    itself would build the wrong flavor and fail on a type it never mentions,
    far from the import that caused it. Catch it here instead.
    """
    root = _ADK_ROOT / _MUST_USE_THE_SEAM
    # A walk over a path that no longer exists finds nothing and passes, which
    # is the one way a tripwire must not fail. Rename the subtree and this
    # says so instead of going quiet.
    assert root.is_dir(), f'{root} is gone; this guard is no longer guarding'

    offenders = {}
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
      dirnames[:] = [d for d in dirnames if d != '__pycache__']
      for filename in filenames:
        if not filename.endswith('.py'):
          continue
        path = pathlib.Path(dirpath) / filename
        scanned += 1
        found = _direct_httpx_imports(path.read_text(encoding='utf-8'))
        if found:
          offenders[path.relative_to(_ADK_ROOT).as_posix()] = found

    assert (
        scanned >= 5
    ), f'only {scanned} files scanned; the walk is not reaching the modules'
    assert not offenders, (
        f'These modules under {_MUST_USE_THE_SEAM} import the HTTP client'
        ' library directly. Import from `google.adk.dependencies._httpx`'
        f' instead: {offenders}'
    )

  def test_every_advertised_name_resolves(self):
    """`__all__` is the contract callers rely on, so it must be honest."""
    missing = [
        name
        for name in httpx_dependency.__all__
        if getattr(httpx_dependency, name, None) is None
    ]

    assert not missing

  def test_the_seam_binds_the_flavor_the_installed_sdk_builds(self):
    """The two have to agree, and only this asserts that they do.

    ADK hands its client to the SDK's transport, so the seam must bind the
    same library the SDK builds its own clients from. Checking the seam
    against the SDK, rather than against a hardcoded name, is what makes this
    fail if the major detection ever keys off the wrong thing.
    """
    sdk_client = mcp_dependency.create_mcp_http_client()
    try:
      assert isinstance(sdk_client, httpx_dependency.AsyncClient)
    finally:
      asyncio.run(sdk_client.aclose())


def _foreign_timeout_class():
  """The other major's `Timeout`, stood in for by the installed one's own code.

  Both majors ship the same constructor, and its first branch asks
  `isinstance(timeout, Timeout)` against a global of the module it was defined
  in. Rebinding that one global to a class of our own is the entire difference
  between a home `Timeout` and a foreign one, so this reaches the foreign
  branches of real library code with only one library installed.
  """
  source = httpx_dependency.Timeout.__init__
  namespace = dict(source.__globals__)
  init = types.FunctionType(
      source.__code__,
      namespace,
      source.__name__,
      source.__defaults__,
      source.__closure__,
  )
  init.__kwdefaults__ = source.__kwdefaults__
  foreign = type('ForeignTimeout', (), {'__init__': init})
  namespace['Timeout'] = foreign
  return foreign


class TestPortableTimeout:
  """A timeout has to survive a factory built on the major ADK did not bind."""

  def _timeout(self):
    return httpx_dependency.Timeout(15.0, read=300.0)

  def test_it_is_a_timeout_of_the_flavor_the_seam_bound(self):
    portable = httpx_dependency.PortableTimeout(self._timeout())

    assert isinstance(portable, httpx_dependency.Timeout)
    assert (portable.connect, portable.read, portable.write, portable.pool) == (
        15.0,
        300.0,
        15.0,
        15.0,
    )

  def test_it_is_also_the_four_item_tuple_the_other_major_falls_back_to(self):
    portable = httpx_dependency.PortableTimeout(self._timeout())

    assert tuple(portable) == (15.0, 300.0, 15.0, 15.0)

  def test_a_constructor_that_does_not_recognize_it_reads_it_anyway(self):
    """This runs whichever major is installed, and one always is.

    It proves the shim against a constructor that does not know its class, not
    against `httpx2` specifically -- that the two majors still dispatch alike
    is what the cross-major test in `test_mcp_session_manager` checks, and that
    one needs both installed.
    """
    foreign_timeout = _foreign_timeout_class()

    # The defect: unrecognized, a plain `Timeout` is stored whole as each field.
    plain = self._timeout()
    assert foreign_timeout(plain).connect is plain

    portable = foreign_timeout(
        httpx_dependency.PortableTimeout(self._timeout())
    )
    assert (portable.connect, portable.read, portable.write, portable.pool) == (
        15.0,
        300.0,
        15.0,
        15.0,
    )
