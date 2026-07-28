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

"""Tests for the FunctionTool declaration cache.

`_get_declaration` is called once per tool on every LLM request, so its result
is cached. Callers are allowed to mutate the declaration they receive, so each
call must still get a private copy, and the cache must follow every input the
declaration depends on.
"""

import copy

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools import _function_tool_declarations
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.adk.utils.variant_utils import GoogleLLMVariant
import pytest


def sample_tool(device_id: str, status: str = 'ON') -> str:
  """Sets a device status.

  Args:
    device_id: The device.
    status: The desired status.

  Returns:
    A confirmation message.
  """
  return 'ok'


def poll_job(job_id: str) -> str:
  """Polls a job.

  Args:
    job_id: The job.

  Returns:
    The job status.
  """
  return 'PENDING'


def test_repeated_calls_return_equal_declarations():
  tool = FunctionTool(sample_tool)

  first = tool._get_declaration()
  second = tool._get_declaration()

  assert first.model_dump() == second.model_dump()


def test_repeated_calls_return_distinct_objects():
  tool = FunctionTool(sample_tool)

  first = tool._get_declaration()
  second = tool._get_declaration()

  assert first is not second
  assert first.parameters_json_schema is not second.parameters_json_schema


def test_declaration_is_built_once_for_repeated_calls():
  tool = FunctionTool(sample_tool)
  calls = []
  original = (
      _function_tool_declarations.build_function_declaration_with_json_schema
  )

  def counting(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)

  _function_tool_declarations.build_function_declaration_with_json_schema = (
      counting
  )
  try:
    for _ in range(5):
      tool._get_declaration()
  finally:
    _function_tool_declarations.build_function_declaration_with_json_schema = (
        original
    )

  assert len(calls) == 1


def test_mutating_a_returned_declaration_does_not_affect_later_calls():
  tool = FunctionTool(sample_tool)

  first = tool._get_declaration()
  first.name = 'renamed'
  first.description = 'mutated'
  first.parameters_json_schema['properties']['device_id']['enum'] = ['x']

  second = tool._get_declaration()

  assert second.name == 'sample_tool'
  assert second.description != 'mutated'
  assert 'enum' not in second.parameters_json_schema['properties']['device_id']


def test_long_running_tool_description_is_stable_across_calls():
  """LongRunningFunctionTool appends a note to the declaration it is given."""
  tool = LongRunningFunctionTool(poll_job)

  descriptions = [tool._get_declaration().description for _ in range(3)]

  assert len(set(descriptions)) == 1
  assert descriptions[0].count('long-running operation') == 1


def test_transfer_to_agent_tool_sets_enum_on_every_call():
  """TransferToAgentTool writes an enum into the parameter schema it is given."""
  tool = TransferToAgentTool(agent_names=['alpha', 'beta'])

  for _ in range(3):
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema['properties']['agent_name'][
        'enum'
    ] == ['alpha', 'beta']


def test_toolset_name_prefixing_does_not_rename_the_original_tool():
  """BaseToolset prefixing writes .name onto the declaration it is given."""
  tool = FunctionTool(sample_tool)
  prefixed = copy.copy(tool)
  original_get_declaration = tool._get_declaration

  def prefixed_declaration():
    declaration = original_get_declaration()
    declaration.name = 'prefix_sample_tool'
    return declaration

  prefixed._get_declaration = prefixed_declaration

  assert prefixed._get_declaration().name == 'prefix_sample_tool'
  assert tool._get_declaration().name == 'sample_tool'


def test_changing_ignore_params_rebuilds_the_declaration():
  tool = FunctionTool(sample_tool)

  before = tool._get_declaration().parameters_json_schema['properties']
  assert 'status' in before

  tool._ignore_params = list(tool._ignore_params) + ['status']
  after = tool._get_declaration().parameters_json_schema['properties']

  assert 'status' not in after


def test_changing_func_rebuilds_the_declaration():
  tool = FunctionTool(sample_tool)
  assert tool._get_declaration().name == 'sample_tool'

  tool.func = poll_job

  assert tool._get_declaration().name == 'poll_job'


@pytest.mark.parametrize(
    'variant',
    [GoogleLLMVariant.GEMINI_API, GoogleLLMVariant.VERTEX_AI],
)
def test_changing_api_variant_rebuilds_the_declaration(monkeypatch, variant):
  """The api variant is resolved from the environment on every call."""
  tool = FunctionTool(sample_tool)

  monkeypatch.setattr(
      type(tool),
      '_api_variant',
      property(lambda self: GoogleLLMVariant.GEMINI_API),
  )
  studio = tool._get_declaration()

  monkeypatch.setattr(
      type(tool), '_api_variant', property(lambda self: variant)
  )
  switched = tool._get_declaration()

  if variant is GoogleLLMVariant.GEMINI_API:
    assert switched.model_dump() == studio.model_dump()
  else:
    assert switched.response_json_schema is not None
    assert studio.response_json_schema is None


@pytest.mark.parametrize(
    'variant',
    [GoogleLLMVariant.GEMINI_API, GoogleLLMVariant.VERTEX_AI],
)
@pytest.mark.parametrize('first_enabled', [False, True])
def test_changing_schema_representation_rebuilds_the_declaration(
    monkeypatch, variant, first_enabled
):
  """The feature flag selects legacy or JSON-schema declaration fields."""
  tool = FunctionTool(sample_tool)
  monkeypatch.setattr(
      type(tool), '_api_variant', property(lambda self: variant)
  )

  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, first_enabled
  ):
    first = tool._get_declaration()
  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, not first_enabled
  ):
    second = tool._get_declaration()

  declarations = {first_enabled: first, not first_enabled: second}
  json_schema = declarations[True]
  legacy = declarations[False]
  assert json_schema.parameters_json_schema is not None
  assert json_schema.parameters is None
  assert legacy.parameters_json_schema is None
  assert legacy.parameters is not None


def test_provider_environment_aliases_rebuild_the_declaration(monkeypatch):
  """Both provider environment variables resolve into the cache's API key."""
  tool = FunctionTool(sample_tool)
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)

  studio = tool._get_declaration()
  monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', '1')
  deprecated_vertex = tool._get_declaration()
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '0')
  enterprise_precedence = tool._get_declaration()
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', 'true')
  enterprise_vertex = tool._get_declaration()

  assert studio.response_json_schema is None
  assert deprecated_vertex.response_json_schema is not None
  assert enterprise_precedence.response_json_schema is None
  assert enterprise_vertex.response_json_schema is not None
