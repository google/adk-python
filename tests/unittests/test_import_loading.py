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

"""Fresh-process checks for ADK's public import-loading contract."""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest import mock

from click.testing import CliRunner
import pytest

from . import isolated_import_utils
from .isolated_import_utils import assert_modules_unloaded
from .isolated_import_utils import run_isolated

pytestmark = pytest.mark.skipif(
    not isolated_import_utils.SOURCE_ROOT.is_dir(),
    reason='Import-loading checks need the source checkout layout.',
)

# Every package whose exports go through google.adk.utils._lazy.accessors.
_LAZY_PACKAGES = (
    'google.adk',
    'google.adk.agents',
    'google.adk.cli',
    'google.adk.cli.utils',
    'google.adk.workflow',
)


@pytest.mark.parametrize(
    ('module_name', 'forbidden'),
    [
        (
            'google.adk',
            (
                'a2a',
                'fastapi',
                'google.adk.agents.llm_agent',
                'google.adk.runners',
                'google.cloud.aiplatform',
                'google.genai',
                'mcp',
                'opentelemetry.sdk',
                'pydantic',
                'sqlalchemy',
                'uvicorn',
            ),
        ),
        (
            'google.adk.agents',
            (
                'google.adk.agents.base_agent',
                'google.adk.agents.llm_agent',
                'google.genai',
                'mcp',
            ),
        ),
        (
            'google.adk.workflow',
            (
                'google.adk.workflow._function_node',
                'google.adk.workflow._join_node',
                'google.adk.workflow._node',
                'google.adk.workflow._workflow',
            ),
        ),
        (
            'google.adk.cli.cli_tools_click',
            (
                'fastapi',
                'google.adk.agents.run_config',
                'google.adk.cli.cli',
                'google.adk.evaluation.agent_evaluator',
                'google.genai',
                'mcp',
                'uvicorn',
            ),
        ),
    ],
    ids=('root', 'agents', 'workflow', 'cli_commands'),
)
def test_package_import_defers_unrelated_runtime(
    module_name: str, forbidden: tuple[str, ...]
) -> None:
  """Importing a lightweight package leaves unrelated runtime stacks alone."""
  assert_modules_unloaded(
      f'import importlib\nimportlib.import_module({module_name!r})', forbidden
  )


def test_constructing_agent_defers_optional_mcp_server_stack():
  """A normal Agent does not import MCP just because its extra is installed."""
  if importlib.util.find_spec('mcp') is None:
    pytest.skip('MCP import-boundary check requires the declared test extra.')

  assert_modules_unloaded(
      """
from google.adk import Agent

Agent(name='agent', model='gemini-2.5-flash')
""",
      ('mcp', 'sse_starlette', 'uvicorn'),
  )


def test_lazy_packages_support_star_imports():
  """Every lazy package still resolves through Python's public import syntax."""
  result = run_isolated(f"""
import importlib

for module_name in {_LAZY_PACKAGES!r}:
  package = importlib.import_module(module_name)
  namespace = {{}}
  exec(f'from {{module_name}} import *', namespace)
  assert set(package.__all__).issubset(dir(package)), module_name
  for name in package.__all__:
    assert namespace[name] is getattr(package, name), (module_name, name)
""")

  assert result.returncode == 0, result.stderr


def test_lazy_packages_resolve_subpackages_as_attributes():
  """A subpackage stays reachable on its parent, as eager imports left it."""
  result = run_isolated("""
import types

import google.adk

for name in ('agents', 'events', 'runners', 'sessions', 'tools'):
  assert isinstance(getattr(google.adk, name), types.ModuleType), name
""")

  assert result.returncode == 0, result.stderr


def test_lazy_packages_reject_unknown_attributes():
  """The lazy hook raises AttributeError rather than masking typos."""
  result = run_isolated(f"""
import importlib

for module_name in {_LAZY_PACKAGES!r}:
  package = importlib.import_module(module_name)
  try:
    package.NotAnExport
  except AttributeError:
    continue
  raise AssertionError(module_name)
""")

  assert result.returncode == 0, result.stderr


def test_conformance_help_keeps_streaming_mode_choices():
  """Conformance help retains the public streaming-mode choices."""
  from google.adk.agents.run_config import StreamingMode
  from google.adk.cli.cli_tools_click import main

  result = CliRunner().invoke(main, ['conformance', 'record', '--help'])
  expected_choices = (
      '{' + '|'.join(str(mode.value).lower() for mode in StreamingMode) + '}'
  )

  assert result.exit_code == 0
  assert expected_choices in result.output.lower()


def test_conformance_record_converts_streaming_mode_at_execution():
  """Conformance execution still receives the runtime streaming enum."""
  from google.adk.agents.run_config import StreamingMode
  from google.adk.cli.cli_tools_click import main

  observed_modes = []

  async def run_conformance_record(_paths, streaming_mode):
    observed_modes.append(streaming_mode)

  module_name = 'google.adk.cli.conformance.cli_record'
  fake_module = types.ModuleType(module_name)
  fake_module.run_conformance_record = run_conformance_record

  with mock.patch.dict(sys.modules, {module_name: fake_module}):
    result = CliRunner().invoke(main, ['conformance', 'record', 'sse'])

  assert result.exit_code == 0, result.exception
  assert observed_modes == [StreamingMode.SSE]
