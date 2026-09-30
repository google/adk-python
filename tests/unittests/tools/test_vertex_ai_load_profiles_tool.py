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

from types import SimpleNamespace

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.vertex_ai_load_profiles_tool import VertexAiLoadProfilesTool
from pytest import mark
from vertexai import types as vertex_types


class _FakeMemoryService:
  """Minimal profile-providing service for VertexAiLoadProfilesTool tests."""

  def __init__(self, profiles):
    self._profiles = profiles
    self.calls = []

  async def retrieve_profiles(self, *, app_name, user_id):
    self.calls.append((app_name, user_id))
    return self._profiles


class _StubToolContext:
  """Minimal ToolContext stub exposing only the scope the tool reads."""

  def __init__(self, *, app_name='test-app', user_id='test-user'):
    self.session = SimpleNamespace(app_name=app_name)
    self.user_id = user_id


@mark.asyncio
async def test_load_profiles_returns_profile_payloads():
  memory_service = _FakeMemoryService([
      vertex_types.MemoryProfile(
          schema_id='user-profile', profile={'name': 'Kim'}
      ),
      vertex_types.MemoryProfile(schema_id='empty', profile={}),
  ])
  tool = VertexAiLoadProfilesTool(memory_service=memory_service)

  result = await tool.load_profiles(_StubToolContext())

  assert result == {'profiles': [{'name': 'Kim'}]}
  assert memory_service.calls == [('test-app', 'test-user')]


def test_get_declaration_with_json_schema_feature_disabled():
  tool = VertexAiLoadProfilesTool(memory_service=_FakeMemoryService([]))
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = tool._get_declaration()

  assert declaration.name == 'load_profiles'
  assert declaration.parameters_json_schema is None
  assert declaration.parameters.properties == {}


def test_get_declaration_with_json_schema_feature_enabled():
  tool = VertexAiLoadProfilesTool(memory_service=_FakeMemoryService([]))
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    declaration = tool._get_declaration()

  assert declaration.name == 'load_profiles'
  assert declaration.parameters is None
  assert declaration.parameters_json_schema == {
      'type': 'object',
      'properties': {},
  }


@mark.asyncio
async def test_process_llm_request_registers_tool_only():
  tool = VertexAiLoadProfilesTool(memory_service=_FakeMemoryService([]))
  llm_request = LlmRequest()

  await tool.process_llm_request(
      tool_context=_StubToolContext(),
      llm_request=llm_request,
  )

  assert llm_request.config.system_instruction is None
  assert llm_request.config.tools is not None
  assert llm_request.config.tools[0].function_declarations is not None
  assert llm_request.config.tools[0].function_declarations[0].name == (
      'load_profiles'
  )
  assert 'load_profiles' in llm_request.tools_dict
