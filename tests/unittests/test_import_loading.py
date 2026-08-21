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
from .isolated_import_utils import loaded_top_level_packages
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

# The statements almost every ADK program starts with, and therefore the two
# import graphs whose cost every user pays.
_ENTRY_POINTS = (
    'from google.adk.agents import Agent',
    'from google.adk.runners import Runner',
)

# Third-party top-level packages an entry point may load. The forbidden lists
# above pin individual deferrals on the lazy package inits; this one bounds the
# whole graph, because the cost that reaches users arrives as a package nobody
# noticed rather than as one somebody predicted.
_ENTRY_POINT_PACKAGE_ALLOWLIST = frozenset({
    # Declared requirements that ADK imports at module scope.
    'click',
    'fastapi',
    'google',
    'httpx',
    'opentelemetry',
    'packaging',
    'pydantic',
    'python_multipart',
    'starlette',
    'tenacity',
    'websockets',
    # Reached through pydantic and httpx rather than through ADK.
    'annotated_doc',
    'annotated_types',
    'anyio',
    'certifi',
    'idna',
    'orjson',
    'pydantic_core',
    'pygments',
    'rich',
    'sniffio',
    'typing_extensions',
    'typing_inspection',
    'zstandard',
    # google.genai.types annotates optional fields with aiohttp and Pillow
    # types and imports whichever of the two the environment happens to have.
    # No ADK module imports either one, so these are absent in some installs.
    'PIL',
    'aiohappyeyeballs',
    'aiohttp',
    'aiosignal',
    'attr',
    'defusedxml',
    'frozenlist',
    'multidict',
    'propcache',
    'yarl',
    # Stand-ins the packages above import below Python 3.11, where the
    # standard library has no equivalent yet: aiohttp takes async_timeout for
    # asyncio.timeout, and anyio takes exceptiongroup for ExceptionGroup.
    'async_timeout',
    'exceptiongroup',
})


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


@pytest.mark.parametrize('statement', _ENTRY_POINTS, ids=('agent', 'runner'))
def test_entry_point_loads_only_allowlisted_packages(statement: str) -> None:
  """The two entry points every program uses load a reviewed set of packages.

  The lazy package inits are already cheap, so a new eager dependency shows up
  here first: as a package nobody agreed to pay for on every ADK start.

  The unit is the top-level import name, so a new eager dependency arriving
  under the `google` namespace, which ADK loads either way, does not show up
  here.
  """
  unexpected = sorted(
      loaded_top_level_packages(statement) - _ENTRY_POINT_PACKAGE_ALLOWLIST
  )

  assert not unexpected, (
      f'{statement!r} now loads {", ".join(unexpected)}, which every ADK'
      ' process would pay for at startup. Move the import into the function'
      ' that needs it, or add the package to the allowlist together with the'
      ' reason it has to be eager.'
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
