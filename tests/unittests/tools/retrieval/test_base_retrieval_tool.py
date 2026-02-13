# Copyright 2025 Google LLC
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

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.genai import types


class _TestRetrievalTool(BaseRetrievalTool):
  """Concrete implementation of BaseRetrievalTool for testing."""

  def __init__(self):
    super().__init__(
        name='test_retrieval',
        description='A test retrieval tool.',
    )

  async def run_async(self, *, args, tool_context):
    return {'result': 'test'}


def test_get_declaration_with_json_schema_feature_disabled():
  """Test that _get_declaration uses parameters when feature is disabled."""
  tool = _TestRetrievalTool()

  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = tool._get_declaration()

  assert declaration.name == 'test_retrieval'
  assert declaration.description == 'A test retrieval tool.'
  assert declaration.parameters_json_schema is None
  assert isinstance(declaration.parameters, types.Schema)
  assert declaration.parameters.type == types.Type.OBJECT
  assert 'query' in declaration.parameters.properties


def test_get_declaration_with_json_schema_feature_enabled():
  """Test that _get_declaration uses parameters_json_schema when feature is enabled."""
  tool = _TestRetrievalTool()

  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    declaration = tool._get_declaration()

  assert declaration.name == 'test_retrieval'
  assert declaration.description == 'A test retrieval tool.'
  assert declaration.parameters is None
  assert declaration.parameters_json_schema == {
      'type': 'object',
      'properties': {
          'query': {
              'type': 'string',
              'description': 'The query to retrieve.',
          },
      },
  }
