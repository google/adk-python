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

import base64

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.load_artifacts_tool import _maybe_base64_to_bytes
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types
from pytest import mark


class _StubToolContext:
  """Minimal ToolContext stub for LoadArtifactsTool tests."""

  def __init__(self, artifacts_by_name: dict[str, types.Part]):
    self._artifacts_by_name = artifacts_by_name

  async def list_artifacts(self) -> list[str]:
    return list(self._artifacts_by_name.keys())

  async def load_artifact(self, name: str) -> types.Part | None:
    return self._artifacts_by_name.get(name)


@mark.asyncio
async def test_load_artifacts_converts_unsupported_mime_to_text():
  """Unsupported inline MIME types are converted to text parts."""
  artifact_name = 'test.csv'
  csv_bytes = b'col1,col2\n1,2\n'
  artifact = types.Part(
      inline_data=types.Blob(data=csv_bytes, mime_type='application/csv')
  )

  tool_context = _StubToolContext({artifact_name: artifact})
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': [artifact_name]},
                      )
                  )
              ],
          )
      ]
  )

  await load_artifacts_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts[0].text == (
      f'Artifact {artifact_name} is:'
  )
  artifact_part = llm_request.contents[-1].parts[1]
  assert artifact_part.inline_data is None
  assert artifact_part.text == csv_bytes.decode('utf-8')


@mark.asyncio
async def test_load_artifacts_converts_base64_unsupported_mime_to_text():
  """Unsupported base64 string data is converted to text parts."""
  artifact_name = 'test.csv'
  csv_bytes = b'col1,col2\n1,2\n'
  csv_base64 = base64.b64encode(csv_bytes).decode('ascii')
  artifact = types.Part(
      inline_data=types.Blob(data=csv_base64, mime_type='application/csv')
  )

  tool_context = _StubToolContext({artifact_name: artifact})
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': [artifact_name]},
                      )
                  )
              ],
          )
      ]
  )

  await load_artifacts_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  artifact_part = llm_request.contents[-1].parts[1]
  assert artifact_part.inline_data is None
  assert artifact_part.text == csv_bytes.decode('utf-8')


@mark.asyncio
async def test_load_artifacts_keeps_supported_mime_types():
  """Supported inline MIME types are passed through unchanged."""
  artifact_name = 'test.pdf'
  artifact = types.Part(
      inline_data=types.Blob(data=b'%PDF-1.4', mime_type='application/pdf')
  )

  tool_context = _StubToolContext({artifact_name: artifact})
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': [artifact_name]},
                      )
                  )
              ],
          )
      ]
  )

  await load_artifacts_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  artifact_part = llm_request.contents[-1].parts[1]
  assert artifact_part.inline_data is not None
  assert artifact_part.inline_data.mime_type == 'application/pdf'


def test_maybe_base64_to_bytes_decodes_standard_base64():
  """Standard base64 encoded strings are decoded correctly."""
  original = b'hello world'
  encoded = base64.b64encode(original).decode('ascii')
  assert _maybe_base64_to_bytes(encoded) == original


def test_maybe_base64_to_bytes_decodes_urlsafe_base64():
  """URL-safe base64 encoded strings are decoded correctly."""
  original = b'\xfb\xff\xfe'  # bytes that produce +/ in std but -_ in urlsafe
  encoded = base64.urlsafe_b64encode(original).decode('ascii')
  assert _maybe_base64_to_bytes(encoded) == original


def test_maybe_base64_to_bytes_returns_none_for_invalid():
  """Invalid base64 strings return None."""
  # Single character is invalid (base64 requires length % 4 == 0 after padding)
  assert _maybe_base64_to_bytes('x') is None


def test_get_declaration_with_json_schema_feature_enabled():
  """Test that _get_declaration uses parameters_json_schema when feature is enabled."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    declaration = load_artifacts_tool._get_declaration()

  assert declaration.name == 'load_artifacts'
  assert declaration.parameters is None
  assert declaration.parameters_json_schema == {
      'type': 'object',
      'properties': {
          'artifact_names': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
  }
