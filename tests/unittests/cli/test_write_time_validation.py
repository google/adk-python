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

"""Tests for the checks that run before Agent Builder writes anything.

Two of these pin regressions that had already shipped. Schema validation was
being bypassed for the most common agent shape, and nothing checked that a
`google.adk.*` name referenced by a generated file actually exists. Both let
broken output reach disk and surface only when the agent failed to load.
"""

from __future__ import annotations

import importlib
from typing import Any

from google.adk.cli.built_in_agents.tools.write_config_files import _validate_adk_symbols
from google.adk.cli.built_in_agents.tools.write_config_files import _validate_against_schema
from google.adk.cli.built_in_agents.tools.write_files import _unknown_adk_imports
from google.adk.cli.built_in_agents.utils._adk_symbols import adk_symbol_exists
import pytest

_LLM_AGENT: dict[str, Any] = {
    'name': 'root_agent',
    'model': 'gemini-2.5-flash',
    'instruction': 'Answer the question.',
}


def _with(**extra: Any) -> dict[str, Any]:
  return {**_LLM_AGENT, **extra}


# --- schema validation ----------------------------------------------------


def test_a_plain_llm_agent_validates():
  """The case the old string-matching bypass existed to paper over.

  AgentConfig's union branches overlap, so a valid LlmAgent matches two of
  them and jsonschema rejects it. Dispatching on agent_class is what makes
  this pass without disabling validation for everything else.
  """
  assert _validate_against_schema(_LLM_AGENT)['valid']


@pytest.mark.parametrize(
    'agent_class', ['SequentialAgent', 'ParallelAgent', 'LoopAgent']
)
def test_workflow_agents_validate(agent_class: str):
  config = {
      'agent_class': agent_class,
      'name': 'root_agent',
      'sub_agents': [{'config_path': 'child.yaml'}],
  }

  assert _validate_against_schema(config)['valid']


def test_a_callback_carrying_args_is_rejected():
  """The regression. CodeConfig takes only `name` in 2.x.

  Before the fix jsonschema surfaced the union ambiguity first, the bypass
  matched on that message, and this was written to disk as valid.
  """
  result = _validate_against_schema(
      _with(
          before_model_callbacks=[
              {'name': 'proj.callbacks.redact', 'args': {'fields': ['ssn']}}
          ]
      )
  )

  assert not result['valid']
  assert result['errors']


def test_args_on_a_tool_is_accepted():
  """`args` is legal on a ToolConfig and only illegal on a CodeConfig.

  Rejecting both would penalise correct configurations.
  """
  config = _with(
      tools=[
          {'name': 'google_search'},
          {'name': 'AgentTool', 'args': {'agent': {'config_path': 'x.yaml'}}},
      ]
  )

  assert _validate_against_schema(config)['valid']


def test_a_config_the_runtime_accepts_is_not_rejected_by_the_schema():
  """The shipped schema is stricter than the models it describes.

  It is emitted with camelCase aliases and additionalProperties false, while
  pydantic also accepts the snake_case names. A live-audio config loads fine
  and must not be refused.
  """
  config = _with(
      generate_content_config={
          'response_modalities': ['AUDIO'],
          'speech_config': {
              'voice_config': {'prebuilt_voice_config': {'voice_name': 'Puck'}}
          },
      }
  )

  assert _validate_against_schema(config)['valid']


def test_a_typo_is_still_rejected():
  assert not _validate_against_schema(
      {'name': 'root_agent', 'model': 'm', 'instrucion': 'typo'}
  )['valid']


def test_a_qualified_agent_class_still_validates_against_its_own_schema():
  """A dotted `agent_class` resolves to the class's own `config_type`.

  Validating through the deprecated `AgentConfig` union routed any non-short
  `agent_class` to `BaseAgentConfig` (`extra='allow'`), which let
  `_runtime_accepts` swallow schema errors whenever `agent_class` was fully
  qualified.
  """
  assert not _validate_against_schema({
      'agent_class': 'google.adk.agents.LlmAgent',
      'name': 'root_agent',
      'model': 'm',
      'instrucion': 'typo',
  })['valid']


# --- symbol resolution ----------------------------------------------------


def test_a_name_that_never_existed_is_rejected():
  errors = _validate_adk_symbols(
      _with(tools=[{'name': 'google.adk.tools.tool_args.ToolArgs'}])
  )

  assert errors
  assert errors[0]['invalid_value'] == 'google.adk.tools.tool_args.ToolArgs'


def test_a_name_removed_since_1_x_is_rejected():
  errors = _validate_adk_symbols(
      _with(
          before_model_callbacks=[
              {'name': 'google.adk.agents.common_configs.ArgumentConfig'}
          ]
      )
  )

  assert errors


@pytest.mark.parametrize(
    'tool_name',
    [
        'google_search',
        'google.adk.tools.google_search',
        'myproject.tools.roll_die',
    ],
)
def test_real_and_local_references_are_accepted(tool_name: str):
  assert not _validate_adk_symbols(_with(tools=[{'name': tool_name}]))


def test_a_name_behind_a_missing_extra_is_accepted():
  """A wheel the developer has not installed is not the model's mistake."""
  assert not _validate_adk_symbols({
      'name': 'root_agent',
      'instruction': 'x',
      'model_code': {'name': 'google.adk.models.lite_llm.LiteLlm'},
  })


def test_resolution_follows_re_exports():
  # Resolving by import rather than against a list of known names is what
  # makes an alias or re-export resolve as the loader will see it.
  assert adk_symbol_exists('google.adk.agents.LlmAgent')
  assert adk_symbol_exists('google.adk.agents.llm_agent.LlmAgent')
  assert not adk_symbol_exists('google.adk.agents.NoSuchAgent')


# --- generated Python -----------------------------------------------------


def test_a_bad_adk_import_is_refused():
  unknown = _unknown_adk_imports(
      'tools/api_tool.py', 'from google.adk.tools.tool_args import ToolArgs\n'
  )

  assert unknown == ['google.adk.tools.tool_args.ToolArgs']


def test_a_good_adk_import_is_allowed():
  assert not _unknown_adk_imports(
      'agent.py',
      'from google.adk.agents import LlmAgent\nimport google.adk.tools\n',
  )


def test_only_imports_are_inspected():
  """An attribute deep in the body may be guarded; an import is a promise."""
  assert not _unknown_adk_imports(
      'agent.py',
      'import google.adk\n\nif False:\n  google.adk.made.up.Thing()\n',
  )


def test_non_python_files_are_left_alone():
  assert not _unknown_adk_imports(
      'root_agent.yaml', 'from google.adk.tools.tool_args import ToolArgs'
  )


def test_a_syntax_error_is_not_reported_as_a_bad_import():
  # Unparseable source is a different complaint, raised elsewhere.
  assert not _unknown_adk_imports('agent.py', 'def broken(:\n')


def test_the_error_names_the_installed_version():
  module = importlib.import_module(
      'google.adk.cli.built_in_agents.tools.write_config_files'
  )

  assert module._adk_version()
