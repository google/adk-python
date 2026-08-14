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
import io
from typing import Any
import zipfile

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.load_artifacts_tool import _maybe_base64_to_bytes
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.genai import types
import pytest


class _StubToolContext:
  """Minimal ToolContext stub for LoadArtifactsTool tests."""

  def __init__(self, artifacts_by_name: dict[str, types.Part]):
    self._artifacts_by_name = artifacts_by_name

  async def list_artifacts(self) -> list[str]:
    return list(self._artifacts_by_name.keys())

  async def load_artifact(self, name: str) -> types.Part | None:
    return self._artifacts_by_name.get(name)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_load_artifacts_converts_csv_octet_stream_to_text():
  """CSV files streamed as octet-stream are extracted using text fallback."""
  artifact_name = 'test.csv'
  csv_bytes = b'col1,col2\n1,2\n'
  artifact = types.Part(
      inline_data=types.Blob(
          data=csv_bytes, mime_type='application/octet-stream'
      )
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


@pytest.mark.asyncio
async def test_load_artifacts_converts_docx_to_text():
  """DOCX binary payloads are extracted to raw text."""
  artifact_name = 'document.docx'

  # Create a minimal valid docx in memory
  docx_bytes_io = io.BytesIO()
  with zipfile.ZipFile(docx_bytes_io, 'w') as zf:
    zf.writestr(
        'word/document.xml',
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document'
        b' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:t>Hello'
        b' DOCX</w:t></w:p></w:body></w:document>',
    )

  docx_bytes = docx_bytes_io.getvalue()

  artifact = types.Part(
      inline_data=types.Blob(
          data=docx_bytes, mime_type='application/octet-stream'
      )
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
  assert artifact_part.text == 'Hello DOCX'


@pytest.mark.asyncio
async def test_load_artifacts_converts_docx_octet_stream_inline_file_to_text():
  """DOCX binary payloads named 'inline-file' with octet-stream are extracted."""
  artifact_name = 'inline-file'

  # Create a minimal valid docx in memory
  docx_bytes_io = io.BytesIO()
  with zipfile.ZipFile(docx_bytes_io, 'w') as zf:
    zf.writestr(
        'word/document.xml',
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document'
        b' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:t>Hello'
        b' Inline DOCX</w:t></w:p></w:body></w:document>',
    )

  docx_bytes = docx_bytes_io.getvalue()

  artifact = types.Part(
      inline_data=types.Blob(
          data=docx_bytes, mime_type='application/octet-stream'
      )
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
  assert artifact_part.text == 'Hello Inline DOCX'


@pytest.mark.asyncio
async def test_load_artifacts_fallback_for_invalid_docx_octet_stream():
  """Invalid DOCX with octet-stream falls back to binary placeholder."""
  artifact_name = 'inline-file'
  invalid_docx_bytes = b'not a zip file'

  artifact = types.Part(
      inline_data=types.Blob(
          data=invalid_docx_bytes, mime_type='application/octet-stream'
      )
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
  assert 'Binary artifact' in artifact_part.text
  assert 'Content cannot be displayed inline' in artifact_part.text


@pytest.mark.asyncio
async def test_load_artifacts_converts_docx_with_custom_namespace_prefix_to_text():
  """DOCX binary payloads with non-standard namespace prefix are extracted."""
  artifact_name = 'document.docx'

  # Create a minimal valid docx in memory with custom namespace prefix 'ns0'
  docx_bytes_io = io.BytesIO()
  with zipfile.ZipFile(docx_bytes_io, 'w') as zf:
    zf.writestr(
        'word/document.xml',
        b'<?xml version="1.0" encoding="UTF-8"'
        b' standalone="yes"?>\n<ns0:document'
        b' xmlns:ns0="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><ns0:body><ns0:p><ns0:t>Hello'
        b' Custom Prefix</ns0:t></ns0:p></ns0:body></ns0:document>',
    )

  docx_bytes = docx_bytes_io.getvalue()

  artifact = types.Part(
      inline_data=types.Blob(
          data=docx_bytes, mime_type='application/octet-stream'
      )
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
  assert artifact_part.text == 'Hello Custom Prefix'


@pytest.mark.asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'mime_type',
    ['image/svg+xml', 'image/svg', 'application/svg+xml', 'image/xml'],
)
async def test_load_artifacts_converts_svg_to_text(mime_type):
  """SVG/XML image variants are rejected by Gemini with 400 INVALID_ARGUMENT,
  so they must fall through to the text-conversion path instead of being
  forwarded as inline image data.
  """
  artifact_name = 'logo.svg'
  svg_bytes = (
      b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
      b'<circle cx="5" cy="5" r="4"/></svg>'
  )
  artifact = types.Part(
      inline_data=types.Blob(data=svg_bytes, mime_type=mime_type)
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
  # The SVG must NOT be forwarded as inline image data — Gemini would 400.
  assert artifact_part.inline_data is None
  # And the original SVG markup is delivered as a text part instead.
  assert artifact_part.text == svg_bytes.decode('utf-8')


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


@pytest.mark.asyncio
async def test_load_artifacts_registers_dynamic_instructions():
  """load_artifacts registers instructions in llm_request._dynamic_instructions."""
  tool_context = _StubToolContext(
      {'doc.txt': types.Part.from_text(text='hello')},
  )
  llm_request = LlmRequest()
  await load_artifacts_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(llm_request._dynamic_instructions) == 1
  assert 'You have a list of artifacts' in llm_request._dynamic_instructions[0]
  assert llm_request.config.system_instruction is None
  assert len(llm_request.contents) == 0


def test_load_artifacts_tool_keyword_only():
  """process_artifact must be passed as keyword argument."""
  with pytest.raises(TypeError):
    LoadArtifactsTool(lambda art, name: art)  # type: ignore[call-arg]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool',
    [
        LoadArtifactsTool(),
        LoadArtifactsTool(process_artifact=None),
        load_artifacts_tool,
    ],
    ids=['default_constructor', 'explicit_none', 'singleton_instance'],
)
async def test_load_artifacts_tool_default_process_artifact(
    tool: LoadArtifactsTool,
):
  """Default LoadArtifactsTool instances use safe conversion for unsupported artifacts."""
  artifact_name = 'data.csv'
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
  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )
  assert len(llm_request.contents) == 2
  assert (
      llm_request.contents[-1].parts[0].text == f'Artifact {artifact_name} is:'
  )
  assert llm_request.contents[-1].parts[1].text == csv_bytes.decode('utf-8')
  assert llm_request.contents[-1].parts[1].inline_data is None


@pytest.mark.asyncio
async def test_load_artifacts_with_custom_process_artifact():
  """Custom process_artifact transforms artifact parts before adding to LLM request."""
  called_args = []

  def custom_filter(artifact: types.Part, artifact_name: str) -> types.Part:
    called_args.append((artifact, artifact_name))
    return types.Part.from_text(
        text=f'Custom transformed content for {artifact_name}'
    )

  tool = LoadArtifactsTool(process_artifact=custom_filter)
  artifact_name = 'data.csv'
  artifact = types.Part(
      inline_data=types.Blob(
          data=b'col1,col2\n1,2\n', mime_type='application/csv'
      )
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

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(called_args) == 1
  assert called_args[0][0] is artifact
  assert called_args[0][1] == artifact_name

  assert len(llm_request.contents) == 2
  assert (
      llm_request.contents[-1].parts[0].text == f'Artifact {artifact_name} is:'
  )
  assert (
      llm_request.contents[-1].parts[1].text
      == f'Custom transformed content for {artifact_name}'
  )


@pytest.mark.asyncio
async def test_load_artifacts_with_async_custom_process_artifact():
  """Async custom process_artifact transforms artifact parts."""
  called_args = []

  async def async_filter(
      artifact: types.Part, artifact_name: str
  ) -> types.Part:
    called_args.append((artifact, artifact_name))
    return types.Part.from_text(
        text=f'Async transformed content for {artifact_name}'
    )

  tool = LoadArtifactsTool(process_artifact=async_filter)
  artifact_name = 'data.csv'
  artifact = types.Part(
      inline_data=types.Blob(
          data=b'col1,col2\n1,2\n', mime_type='application/csv'
      )
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

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(called_args) == 1
  assert called_args[0][0] is artifact
  assert called_args[0][1] == artifact_name

  assert len(llm_request.contents) == 2
  assert (
      llm_request.contents[-1].parts[0].text == f'Artifact {artifact_name} is:'
  )
  assert (
      llm_request.contents[-1].parts[1].text
      == f'Async transformed content for {artifact_name}'
  )


@pytest.mark.asyncio
async def test_load_artifacts_custom_process_artifact_exception_skipped():
  """When process_artifact raises an exception, it is logged and the artifact is skipped."""

  def failing_filter(artifact: types.Part, artifact_name: str) -> types.Part:
    if artifact_name == 'error.txt':
      raise ValueError('Transformation failed!')
    return artifact

  tool = LoadArtifactsTool(process_artifact=failing_filter)
  art1 = types.Part.from_text(text='error content')
  art2 = types.Part.from_text(text='good content')
  tool_context = _StubToolContext({
      'error.txt': art1,
      'good.txt': art2,
  })
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={
                              'artifact_names': ['error.txt', 'good.txt']
                          },
                      )
                  )
              ],
          )
      ]
  )

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(llm_request.contents) == 2
  assert llm_request.contents[1].parts[0].text == 'Artifact good.txt is:'
  assert llm_request.contents[1].parts[1] is art2


@pytest.mark.asyncio
async def test_load_artifacts_custom_filter_multiple_artifacts():
  """Custom filter processes multiple requested artifacts independently."""

  def custom_filter(artifact: types.Part, artifact_name: str) -> types.Part:
    if artifact_name.endswith('.txt'):
      return types.Part.from_text(text=f'PROCESSED_TXT: {artifact.text}')
    return types.Part.from_text(text=f'PROCESSED_OTHER: {artifact_name}')

  tool = LoadArtifactsTool(process_artifact=custom_filter)
  art1 = types.Part.from_text(text='hello world')
  art2 = types.Part(
      inline_data=types.Blob(data=b'%PDF-1.4', mime_type='application/pdf')
  )
  tool_context = _StubToolContext({
      'notes.txt': art1,
      'doc.pdf': art2,
  })
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': ['notes.txt', 'doc.pdf']},
                      )
                  )
              ],
          )
      ]
  )

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(llm_request.contents) == 3
  assert llm_request.contents[1].parts[0].text == 'Artifact notes.txt is:'
  assert llm_request.contents[1].parts[1].text == 'PROCESSED_TXT: hello world'
  assert llm_request.contents[2].parts[0].text == 'Artifact doc.pdf is:'
  assert llm_request.contents[2].parts[1].text == 'PROCESSED_OTHER: doc.pdf'


@pytest.mark.asyncio
async def test_load_artifacts_custom_filter_passthrough_selected_artifacts():
  """Custom filter can pass through selected artifacts unchanged."""

  def custom_filter(artifact: types.Part, artifact_name: str) -> types.Part:
    if artifact_name == 'custom.txt':
      return types.Part.from_text(text='rewritten')
    return artifact

  tool = LoadArtifactsTool(process_artifact=custom_filter)
  art1 = types.Part.from_text(text='original')
  art2 = types.Part.from_text(text='untouched')
  tool_context = _StubToolContext({
      'custom.txt': art1,
      'other.txt': art2,
  })
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={
                              'artifact_names': ['custom.txt', 'other.txt']
                          },
                      )
                  )
              ],
          )
      ]
  )

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(llm_request.contents) == 3
  assert llm_request.contents[1].parts[1].text == 'rewritten'
  assert llm_request.contents[2].parts[1] is art2


@pytest.mark.asyncio
async def test_load_artifacts_custom_filter_returns_none_skips_artifact():
  """When custom filter returns None, the artifact is skipped and omitted from contents."""

  def custom_filter(
      artifact: types.Part, artifact_name: str
  ) -> types.Part | None:
    if artifact_name == 'skip.txt':
      return None
    return artifact

  tool = LoadArtifactsTool(process_artifact=custom_filter)
  art1 = types.Part.from_text(text='skip me')
  art2 = types.Part.from_text(text='keep me')
  tool_context = _StubToolContext({
      'skip.txt': art1,
      'keep.txt': art2,
  })
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': ['skip.txt', 'keep.txt']},
                      )
                  )
              ],
          )
      ]
  )

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Only keep.txt should be appended; skip.txt should be omitted.
  assert len(llm_request.contents) == 2
  assert llm_request.contents[1].parts[0].text == 'Artifact keep.txt is:'
  assert llm_request.contents[1].parts[1] is art2


@pytest.mark.asyncio
async def test_load_artifacts_custom_callback_user_prefixed_fallback():
  """Custom callback receives the unprefixed artifact name during user: fallback."""
  called_args = []

  def custom_filter(artifact: types.Part, artifact_name: str) -> types.Part:
    called_args.append((artifact, artifact_name))
    return types.Part.from_text(text=f'Transformed {artifact_name}')

  tool = LoadArtifactsTool(process_artifact=custom_filter)
  artifact = types.Part.from_text(text='user-scoped data')
  tool_context = _StubToolContext({'user:doc.txt': artifact})
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': ['doc.txt']},
                      )
                  )
              ],
          )
      ]
  )

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(called_args) == 1
  assert called_args[0][0] is artifact
  assert called_args[0][1] == 'doc.txt'

  assert len(llm_request.contents) == 2
  assert llm_request.contents[1].parts[0].text == 'Artifact doc.txt is:'
  assert llm_request.contents[1].parts[1].text == 'Transformed doc.txt'


@pytest.mark.asyncio
async def test_load_artifacts_custom_callback_returns_non_part_raises():
  """When custom callback returns a non-Part object, an error is raised when building Content."""

  def invalid_filter(artifact: types.Part, artifact_name: str) -> Any:
    del artifact, artifact_name
    return 12345

  tool = LoadArtifactsTool(process_artifact=invalid_filter)  # type: ignore[arg-type]
  artifact = types.Part.from_text(text='content')
  tool_context = _StubToolContext({'data.txt': artifact})
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='load_artifacts',
                          response={'artifact_names': ['data.txt']},
                      )
                  )
              ],
          )
      ]
  )

  with pytest.raises((ValueError, TypeError)):
    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
