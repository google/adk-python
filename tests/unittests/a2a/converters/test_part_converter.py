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
from unittest.mock import patch

from a2a import types as a2a_types
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_END_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_START_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_TEXT_MIME_TYPE
from google.adk.a2a.converters.part_converter import _part_data_as_dict
from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.genai import types as genai_types
from google.protobuf import json_format
import pytest


def _make_data_part(data: dict, metadata: dict | None = None) -> a2a_types.Part:
  """Helper to create a proto Part with the data oneof field."""
  part = a2a_types.Part()
  json_format.ParseDict({'data': data}, part)
  if metadata:
    for k, v in metadata.items():
      part.metadata[k] = v
  return part


class TestConvertA2aPartToGenaiPart:
  """Test cases for convert_a2a_part_to_genai_part function."""

  def test_convert_text_part(self):
    """Text Part converts to genai Part with the same text."""
    a2a_part = a2a_types.Part(text="Hello, world!")

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.text == "Hello, world!"

  def test_convert_file_part_with_uri(self):
    """URL Part converts to genai Part with file_data."""
    a2a_part = a2a_types.Part(
        url="gs://bucket/file.txt",
        media_type="text/plain",
        filename="my_file.txt",
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.file_data is not None
    assert result.file_data.file_uri == "gs://bucket/file.txt"
    assert result.file_data.mime_type == "text/plain"
    assert result.file_data.display_name == "my_file.txt"

  def test_convert_file_part_with_bytes(self):
    """Raw bytes Part converts to genai Part with inline_data."""
    test_bytes = b"test file content"
    a2a_part = a2a_types.Part(
        raw=test_bytes,
        media_type="text/plain",
        filename="my_bytes.txt",
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.inline_data is not None
    assert result.inline_data.data == test_bytes
    assert result.inline_data.mime_type == "text/plain"
    assert result.inline_data.display_name == "my_bytes.txt"

  def test_convert_data_part_function_call(self):
    """Data Part with function_call metadata converts to genai FunctionCall Part."""
    function_call_data = {
        "name": "test_function",
        "args": {"param1": "value1"},
    }
    a2a_part = _make_data_part(
        function_call_data,
        {_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL},
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_call is not None
    assert result.function_call.name == "test_function"
    assert result.function_call.args == {"param1": "value1"}

  def test_convert_data_part_function_response(self):
    """Data Part with function_response metadata converts to genai FunctionResponse Part."""
    function_response_data = {
        "name": "test_function",
        "response": {"result": "success", "data": [1, 2, 3]},
    }
    a2a_part = _make_data_part(
        function_response_data,
        {_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE},
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_response is not None
    assert result.function_response.name == "test_function"

  def test_convert_data_part_to_inline_data(self):
    """Data Part without special metadata falls back to a tagged inline blob."""
    data = {"key": "value", "number": 123}
    a2a_part = _make_data_part(data)

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.inline_data is not None
    assert result.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
    assert result.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
    assert result.inline_data.data.endswith(A2A_DATA_PART_END_TAG)

  def test_convert_unsupported_part_type_returns_none(self):
    """An empty Part (no content oneof set) returns None with a warning."""
    a2a_part = a2a_types.Part()  # no content field set

    with patch("google.adk.a2a.converters.part_converter.logger") as mock_logger:
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is None
    mock_logger.warning.assert_called_once()


class TestConvertGenaiPartToA2aPart:
  """Test cases for convert_genai_part_to_a2a_part function."""

  def test_convert_text_part(self):
    """Genai text Part converts to A2A text Part."""
    genai_part = genai_types.Part(text="Hello, world!")

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert result.WhichOneof('content') == 'text'
    assert result.text == "Hello, world!"

  def test_convert_text_part_with_thought(self):
    """Genai text Part with thought=True stores thought in metadata."""
    genai_part = genai_types.Part(text="Hello, world!", thought=True)

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'text'
    assert result.text == "Hello, world!"
    thought_key = _get_adk_metadata_key("thought")
    assert thought_key in result.metadata and result.metadata[thought_key]

  def test_convert_empty_text_part(self):
    """Empty-string text part is preserved, not dropped."""
    genai_part = genai_types.Part(text="")

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'text'
    assert result.text == ""

  def test_convert_file_data_part(self):
    """Genai file_data Part converts to A2A url Part."""
    genai_part = genai_types.Part(
        file_data=genai_types.FileData(
            file_uri="gs://bucket/file.txt",
            mime_type="text/plain",
            display_name="my_file.txt",
        )
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'url'
    assert result.url == "gs://bucket/file.txt"
    assert result.media_type == "text/plain"
    assert result.filename == "my_file.txt"

  def test_convert_inline_data_part(self):
    """Genai inline_data Part converts to A2A raw Part."""
    test_bytes = b"test file content"
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=test_bytes,
            mime_type="text/plain",
            display_name="my_bytes.txt",
        )
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'raw'
    assert result.raw == test_bytes
    assert result.media_type == "text/plain"
    assert result.filename == "my_bytes.txt"

  def test_convert_inline_data_part_with_video_metadata(self):
    """Genai inline_data with video_metadata stores the metadata."""
    test_bytes = b"test video content"
    video_metadata = genai_types.VideoMetadata(fps=30.0)
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(data=test_bytes, mime_type="video/mp4"),
        video_metadata=video_metadata,
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'raw'
    assert _get_adk_metadata_key("video_metadata") in result.metadata

  def test_convert_inline_data_part_to_data_part(self):
    """Tagged blob inline_data round-trips back to a data Part."""
    data = {"key": "value"}
    original = _make_data_part(data)
    original_json = json_format.MessageToJson(original).encode("utf-8")
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=A2A_DATA_PART_START_TAG + original_json + A2A_DATA_PART_END_TAG,
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert _part_data_as_dict(result) == data

  def test_convert_function_call_part(self):
    """Genai function_call Part converts to A2A data Part with function_call metadata."""
    function_call = genai_types.FunctionCall(
        name="test_function", args={"param1": "value1", "param2": 42}
    )
    genai_part = genai_types.Part(function_call=function_call)

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert (
        result.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )
    data = _part_data_as_dict(result)
    assert data["name"] == "test_function"

  def test_convert_function_response_part(self):
    """Genai function_response Part converts to A2A data Part with function_response metadata."""
    function_response = genai_types.FunctionResponse(
        name="test_function", response={"result": "success"}
    )
    genai_part = genai_types.Part(function_response=function_response)

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert (
        result.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
    )

  def test_convert_code_execution_result_part(self):
    """Genai code_execution_result Part converts to A2A data Part."""
    code_execution_result = genai_types.CodeExecutionResult(
        outcome=genai_types.Outcome.OUTCOME_OK, output="Hello, World!"
    )
    genai_part = genai_types.Part(code_execution_result=code_execution_result)

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert (
        result.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
        == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
    )

  def test_convert_executable_code_part(self):
    """Genai executable_code Part converts to A2A data Part."""
    executable_code = genai_types.ExecutableCode(
        language=genai_types.Language.PYTHON, code="print('Hello')"
    )
    genai_part = genai_types.Part(executable_code=executable_code)

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert (
        result.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
        == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
    )

  def test_convert_unsupported_part(self):
    """An empty genai Part returns None with a warning."""
    genai_part = genai_types.Part()

    with patch("google.adk.a2a.converters.part_converter.logger") as mock_logger:
      result = convert_genai_part_to_a2a_part(genai_part)

    assert result is None
    mock_logger.warning.assert_called_once()


class TestRoundTripConversions:
  """Round-trip conversions preserve data through both directions."""

  def test_text_part_round_trip(self):
    """Text part survives A2A → GenAI → A2A round trip."""
    original_text = "Hello, world!"
    a2a_part = a2a_types.Part(text=original_text)

    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'text'
    assert result.text == original_text

  def test_text_part_with_thought_round_trip(self):
    """Text part with thought survives GenAI → A2A → GenAI round trip."""
    genai_part = genai_types.Part(text="Thinking...", thought=True)

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.text == "Thinking..."
    assert result.thought

  def test_file_uri_round_trip(self):
    """URL part survives A2A → GenAI → A2A round trip."""
    a2a_part = a2a_types.Part(
        url="gs://bucket/file.txt",
        media_type="text/plain",
    )

    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'url'
    assert result.url == "gs://bucket/file.txt"
    assert result.media_type == "text/plain"

  def test_file_bytes_round_trip(self):
    """Bytes part survives GenAI → A2A → GenAI round trip."""
    original_bytes = b"test file content for round trip"
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(data=original_bytes, mime_type="application/octet-stream")
    )

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.inline_data is not None
    assert result.inline_data.data == original_bytes

  def test_function_call_round_trip(self):
    """Function call part survives GenAI → A2A → GenAI round trip."""
    function_call = genai_types.FunctionCall(
        name="test_function", args={"param1": "value1", "param2": 42}
    )
    genai_part = genai_types.Part(function_call=function_call)

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_call is not None
    assert result.function_call.name == "test_function"
    assert result.function_call.args == {"param1": "value1", "param2": 42}

  def test_function_response_round_trip(self):
    """Function response part survives GenAI → A2A → GenAI round trip."""
    function_response = genai_types.FunctionResponse(
        name="test_function", response={"result": "success", "data": [1, 2, 3]}
    )
    genai_part = genai_types.Part(function_response=function_response)

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_response is not None
    assert result.function_response.name == "test_function"

  def test_code_execution_result_round_trip(self):
    """Code execution result part survives GenAI → A2A → GenAI round trip."""
    cer = genai_types.CodeExecutionResult(
        outcome=genai_types.Outcome.OUTCOME_OK, output="Hello, World!"
    )
    genai_part = genai_types.Part(code_execution_result=cer)

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.code_execution_result is not None
    assert result.code_execution_result.outcome == cer.outcome
    assert result.code_execution_result.output == cer.output

  def test_executable_code_round_trip(self):
    """Executable code part survives GenAI → A2A → GenAI round trip."""
    ec = genai_types.ExecutableCode(
        language=genai_types.Language.PYTHON, code="print('Hello')"
    )
    genai_part = genai_types.Part(executable_code=ec)

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.executable_code is not None
    assert result.executable_code.language == ec.language
    assert result.executable_code.code == ec.code

  def test_data_part_round_trip(self):
    """Data part survives A2A → GenAI → A2A round trip via tagged blob."""
    data = {"key": "value"}
    a2a_part = _make_data_part(data)

    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    assert _part_data_as_dict(result) == data

  def test_text_part_metadata_round_trip(self):
    """Test round-trip conversion for text parts with metadata."""
    # Arrange
    metadata = {"key1": "value1", "key2": "value2"}
    a2a_part = a2a_types.Part(text="some text")
    a2a_part.metadata.update(metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert result_a2a_part.WhichOneof("content") == "text"
    assert result_a2a_part.text == "some text"
    assert result_a2a_part.metadata["key1"] == "value1"
    assert result_a2a_part.metadata["key2"] == "value2"

  def test_file_part_metadata_round_trip(self):
    """Test round-trip conversion for file parts with metadata."""
    # Arrange
    metadata = {"key1": "value1"}
    a2a_part = a2a_types.Part(
        url="gs://bucket/file.txt",
        media_type="text/plain",
        filename="my_file.txt",
    )
    a2a_part.metadata.update(metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert result_a2a_part.WhichOneof("content") == "url"
    assert result_a2a_part.url == "gs://bucket/file.txt"
    assert result_a2a_part.metadata["key1"] == "value1"


class TestEdgeCases:
  """Edge cases and error conditions."""

  def test_empty_text_part(self):
    """Empty string text part converts successfully."""
    a2a_part = a2a_types.Part(text="")

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.text == ""

  def test_none_input_a2a_to_genai_raises(self):
    """None input to A2A converter raises AttributeError."""
    with pytest.raises(AttributeError):
      convert_a2a_part_to_genai_part(None)

  def test_none_input_genai_to_a2a_raises(self):
    """None input to GenAI converter raises AttributeError."""
    with pytest.raises(AttributeError):
      convert_genai_part_to_a2a_part(None)


class TestNewConstants:
  """Constants exported from part_converter are correct."""

  def test_new_constants_exist(self):
    """Code execution result and executable code constants are defined."""
    assert A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT == "code_execution_result"
    assert A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE == "executable_code"

  def test_convert_a2a_data_part_with_code_execution_result_metadata(self):
    """Data Part with code_execution_result metadata yields a CodeExecutionResult part."""
    a2a_part = _make_data_part(
        {"outcome": "OUTCOME_OK", "output": "Hello, World!"},
        {_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT},
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.code_execution_result is not None
    assert result.code_execution_result.outcome == genai_types.Outcome.OUTCOME_OK
    assert result.code_execution_result.output == "Hello, World!"

  def test_convert_a2a_data_part_with_executable_code_metadata(self):
    """Data Part with executable_code metadata yields an ExecutableCode part."""
    a2a_part = _make_data_part(
        {"language": "PYTHON", "code": "print('Hello')"},
        {_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE},
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.executable_code is not None
    assert result.executable_code.language == genai_types.Language.PYTHON


class TestThoughtSignaturePreservation:
  """thought_signature is preserved through conversions."""

  def test_genai_function_call_with_thought_signature_to_a2a(self):
    """thought_signature is base64-encoded into metadata during GenAI → A2A."""
    function_call = genai_types.FunctionCall(
        id="fc_gemini3", name="my_tool", args={"document": "test"}
    )
    genai_part = genai_types.Part(
        function_call=function_call,
        thought_signature=b"gemini3_signature_bytes",
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    assert result.WhichOneof('content') == 'data'
    thought_sig_key = _get_adk_metadata_key("thought_signature")
    assert thought_sig_key in result.metadata
    assert (
        base64.b64decode(result.metadata[thought_sig_key])
        == b"gemini3_signature_bytes"
    )

  def test_genai_function_call_without_thought_signature_to_a2a(self):
    """Function call without thought_signature doesn't set the metadata key."""
    genai_part = genai_types.Part(
        function_call=genai_types.FunctionCall(id="fc", name="tool", args={})
    )

    result = convert_genai_part_to_a2a_part(genai_part)

    assert result is not None
    thought_sig_key = _get_adk_metadata_key("thought_signature")
    assert thought_sig_key not in result.metadata

  def test_a2a_function_call_with_thought_signature_to_genai(self):
    """Base64-encoded thought_signature in metadata is decoded during A2A → GenAI."""
    sig_b64 = base64.b64encode(b"restored_signature").decode("utf-8")
    a2a_part = _make_data_part(
        {"id": "fc_gemini3", "name": "my_tool", "args": {}},
        {
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            _get_adk_metadata_key("thought_signature"): sig_b64,
        },
    )

    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_call is not None
    assert result.thought_signature == b"restored_signature"

  def test_function_call_with_thought_signature_round_trip(self):
    """thought_signature is preserved in GenAI → A2A → GenAI round trip."""
    original_signature = b"round_trip_signature_test"
    genai_part = genai_types.Part(
        function_call=genai_types.FunctionCall(id="fc", name="tool", args={"key": "val"}),
        thought_signature=original_signature,
    )

    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.thought_signature == original_signature
