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
import json
from unittest.mock import Mock
from unittest.mock import patch

from a2a import types as a2a_types
from google.adk.a2a import _compat
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_END_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_START_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_TEXT_MIME_TYPE
from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types as genai_types
import pytest


def _normalize_numbers(value):
  """Recursively coerce ints to floats so int-vs-float compares are tolerant.

  On a2a-sdk 1.x, structured data round-trips through a protobuf Struct, which
  stores every number as a float. This helper lets assertions compare data dicts
  regardless of whether numbers come back as ``int`` (0.3.x) or ``float`` (1.x).
  """
  if isinstance(value, bool):
    return value
  if isinstance(value, int):
    return float(value)
  if isinstance(value, dict):
    return {k: _normalize_numbers(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_normalize_numbers(v) for v in value]
  return value


class TestConvertA2aPartToGenaiPart:
  """Test cases for convert_a2a_part_to_genai_part function."""

  def test_convert_text_part(self):
    """Test conversion of A2A TextPart to GenAI Part."""
    # Arrange
    a2a_part = _compat.make_text_part("Hello, world!")

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.text == "Hello, world!"

  def test_convert_file_part_with_uri(self):
    """Test conversion of A2A FilePart with URI to GenAI Part."""
    # Arrange
    a2a_part = _compat.make_file_part_with_uri(
        uri="gs://bucket/file.txt",
        mime_type="text/plain",
        name="my_file.txt",
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.file_data is not None
    assert result.file_data.file_uri == "gs://bucket/file.txt"
    assert result.file_data.mime_type == "text/plain"
    assert result.file_data.display_name == "my_file.txt"

  def test_convert_file_part_with_bytes(self):
    """Test conversion of A2A FilePart with bytes to GenAI Part."""
    # Arrange
    test_bytes = b"test file content"
    a2a_part = _compat.make_file_part_with_bytes(
        data=test_bytes,
        mime_type="text/plain",
        name="my_bytes.txt",
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.inline_data is not None
    # The converter decodes base64 back to original bytes
    assert result.inline_data.data == test_bytes
    assert result.inline_data.mime_type == "text/plain"
    assert result.inline_data.display_name == "my_bytes.txt"

  def test_convert_data_part_function_call(self):
    """Test conversion of A2A DataPart with function call metadata."""
    # Arrange
    function_call_data = {
        "name": "test_function",
        "args": {"param1": "value1", "param2": 42},
    }
    a2a_part = _compat.make_data_part(
        data=function_call_data,
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            "adk_type": A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.function_call is not None
    assert result.function_call.name == "test_function"
    assert result.function_call.args == {"param1": "value1", "param2": 42}

  def test_convert_data_part_function_response(self):
    """Test conversion of A2A DataPart with function response metadata."""
    # Arrange
    function_response_data = {
        "name": "test_function",
        "response": {"result": "success", "data": [1, 2, 3]},
    }
    a2a_part = _compat.make_data_part(
        data=function_response_data,
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
            "adk_type": A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.function_response is not None
    assert result.function_response.name == "test_function"
    assert result.function_response.response == {
        "result": "success",
        "data": [1, 2, 3],
    }

  @pytest.mark.parametrize(
      "test_name, data, metadata",
      [
          (
              "without_special_metadata",
              {"key": "value", "number": 123},
              {"other": "metadata"},
          ),
          (
              "no_metadata",
              {"key": "value", "array": [1, 2, 3]},
              None,
          ),
          (
              "complex_data",
              {
                  "nested": {
                      "array": [1, 2, {"inner": "value"}],
                      "boolean": True,
                      "null_value": None,
                  },
                  "unicode": "Hello 世界 🌍",
              },
              None,
          ),
          (
              "empty_metadata",
              {"key": "value"},
              {},
          ),
      ],
  )
  def test_convert_data_part_to_inline_data(self, test_name, data, metadata):
    """Test conversion of A2A DataPart to GenAI inline_data Part."""
    # Arrange
    a2a_part = _compat.make_data_part(data=data, metadata=metadata)

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.inline_data is not None
    assert result.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
    assert result.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
    assert result.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    # The embedded payload is the serialized data part; parse it directly so
    # the assertion is version-agnostic. The shapes differ between SDKs:
    #   0.3.x: the whole DataPart is serialized -> {"data": ..., "metadata":
    #          ..., "kind": "data"}.
    #   1.x:   only the structured ``data`` dict is serialized; metadata is
    #          carried on the genai part (``part_metadata``) instead.
    embedded = result.inline_data.data[
        len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
    ]
    converted_data_part = json.loads(embedded)
    if _compat.IS_A2A_V1:
      embedded_data = converted_data_part
    else:
      embedded_data = converted_data_part["data"]
      # ``metadata`` may be omitted, None, or empty depending on the input;
      # treat all empty forms as equivalent.
      actual_metadata = converted_data_part.get("metadata") or None
      expected_metadata = _normalize_numbers(metadata) if metadata else None
      assert _normalize_numbers(actual_metadata) == expected_metadata
    # On 1.x protobuf Struct stores all numbers as floats; normalize both
    # sides so int-vs-float differences don't fail the comparison.
    assert _normalize_numbers(embedded_data) == _normalize_numbers(data)

  @pytest.mark.skipif(_compat.IS_A2A_V1, reason="0.3-only .root dispatch")
  def test_convert_unsupported_file_type(self):
    """Test handling of unsupported file types."""

    # Arrange - Create a mock unsupported file type
    class UnsupportedFileType:
      pass

    # Create a part manually since FilePart validation might reject it
    mock_file_part = Mock()
    mock_file_part.file = UnsupportedFileType()
    a2a_part = Mock()
    a2a_part.root = mock_file_part

    # Act
    with patch(
        "google.adk.a2a.converters.part_converter.logger"
    ) as mock_logger:
      result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is None
    mock_logger.warning.assert_called_once()

  @pytest.mark.skipif(_compat.IS_A2A_V1, reason="0.3-only .root dispatch")
  def test_convert_unsupported_part_type(self):
    """Test handling of unsupported part types."""

    # Arrange - Create a mock unsupported part type
    class UnsupportedPartType:
      pass

    mock_part = Mock()
    mock_part.root = UnsupportedPartType()

    # Act
    with patch(
        "google.adk.a2a.converters.part_converter.logger"
    ) as mock_logger:
      result = convert_a2a_part_to_genai_part(mock_part)

    # Assert
    assert result is None
    mock_logger.warning.assert_called_once()


class TestConvertA2aPartToGenaiPartApiVariant:
  """Tests for part_metadata suppression based on api_variant (Vertex AI)."""

  def _text_part_with_metadata(self):
    part = _compat.make_text_part("hello")
    _compat.set_part_metadata(
        part,
        {
            _get_adk_metadata_key("thought"): True,
            "custom": "value",
        },
    )
    return part

  def test_text_part_metadata_suppressed_in_vertex_mode(self):
    """In Vertex AI mode, part_metadata must be None to avoid SDK ValueError."""
    a2a_part = self._text_part_with_metadata()

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.VERTEX_AI,
    ):
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.part_metadata is None
    # Native fields are still populated from the metadata.
    assert result.text == "hello"
    assert result.thought is True

  def test_text_part_metadata_preserved_in_gemini_api_mode(self):
    """In Gemini Developer API mode, part_metadata is preserved."""
    a2a_part = self._text_part_with_metadata()

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.GEMINI_API,
    ):
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.part_metadata == {
        _get_adk_metadata_key("thought"): True,
        "custom": "value",
    }

  def test_function_call_metadata_suppressed_in_vertex_mode(self):
    """Function call data parts also suppress part_metadata in Vertex mode."""
    a2a_part = _compat.make_data_part(
        data={"name": "my_func", "args": {"x": 1}},
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            "custom": "value",
        },
    )

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.VERTEX_AI,
    ):
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_call is not None
    assert result.part_metadata is None

  def test_function_response_metadata_suppressed_in_vertex_mode(self):
    """Function response data parts suppress part_metadata in Vertex mode."""
    a2a_part = _compat.make_data_part(
        data={"name": "my_func", "response": {"ok": True}},
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
            "custom": "value",
        },
    )

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.VERTEX_AI,
    ):
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_response is not None
    assert result.part_metadata is None

  def test_file_with_uri_metadata_suppressed_in_vertex_mode(self):
    """File parts suppress part_metadata in Vertex mode."""
    a2a_part = _compat.make_file_part_with_uri(
        uri="gs://bucket/file.txt",
        mime_type="text/plain",
        name="my_file.txt",
    )
    _compat.set_part_metadata(a2a_part, {"custom": "value"})

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.VERTEX_AI,
    ):
      result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.file_data is not None
    assert result.part_metadata is None

  def test_api_variant_resolved_from_env(self):
    """The api variant is resolved via get_google_llm_variant."""
    a2a_part = self._text_part_with_metadata()

    with patch(
        "google.adk.a2a.converters.part_converter.get_google_llm_variant",
        return_value=GoogleLLMVariant.VERTEX_AI,
    ) as mock_get_variant:
      result = convert_a2a_part_to_genai_part(a2a_part)

    mock_get_variant.assert_called_once()
    assert result is not None
    assert result.part_metadata is None


class TestConvertGenaiPartToA2aPart:
  """Test cases for convert_genai_part_to_a2a_part function."""

  def test_convert_text_part(self):
    """Test conversion of GenAI text Part to A2A Part."""
    # Arrange
    genai_part = genai_types.Part(text="Hello, world!")

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_text_part(result)
    assert _compat.part_text(result) == "Hello, world!"

  def test_convert_text_part_with_thought(self):
    """Test conversion of GenAI text Part with thought to A2A Part."""
    # Arrange - thought is a boolean field in genai_types.Part
    genai_part = genai_types.Part(text="Hello, world!", thought=True)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_text_part(result)
    assert _compat.part_text(result) == "Hello, world!"
    metadata = _compat.part_metadata(result)
    assert metadata
    assert metadata[_get_adk_metadata_key("thought")]

  def test_convert_empty_text_part(self):
    """Test that Part(text='') is preserved, not dropped.

    Regression test for #5341: empty-string text parts are valid and
    must not fall through to the unsupported-part warning.
    """
    # Arrange
    genai_part = genai_types.Part(text="")

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert — should produce a valid TextPart, not None
    assert result is not None
    assert _compat.is_text_part(result)
    assert _compat.part_text(result) == ""

  def test_convert_file_data_part(self):
    """Test conversion of GenAI file_data Part to A2A Part."""
    # Arrange
    genai_part = genai_types.Part(
        file_data=genai_types.FileData(
            file_uri="gs://bucket/file.txt",
            mime_type="text/plain",
            display_name="my_file.txt",
        )
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_file_part(result)
    assert _compat.file_part_uri(result) == "gs://bucket/file.txt"
    assert _compat.file_part_mime_type(result) == "text/plain"
    assert _compat.file_part_name(result) == "my_file.txt"

  def test_convert_inline_data_part(self):
    """Test conversion of GenAI inline_data Part to A2A Part."""
    # Arrange
    test_bytes = b"test file content"
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=test_bytes,
            mime_type="text/plain",
            display_name="my_bytes.txt",
        )
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_file_part(result)
    # The version-agnostic reader returns the raw (decoded) bytes on both SDKs.
    assert _compat.file_part_bytes(result) == test_bytes
    assert _compat.file_part_mime_type(result) == "text/plain"
    # Filename is preserved on both SDKs.
    assert _compat.file_part_name(result) == "my_bytes.txt"

  def test_convert_inline_data_part_empty_blob_is_skipped(self):
    """A degenerate inline_data Blob with no payload is unconvertible (None)."""
    # Arrange: an empty Blob has ``data is None``.
    genai_part = genai_types.Part(inline_data=genai_types.Blob())

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is None

  def test_convert_inline_data_part_empty_bytes_is_kept(self):
    """A Blob with present-but-empty bytes (b"") is still converted."""
    # Arrange
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(data=b"", mime_type="text/plain")
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert _compat.is_file_part(result)
    assert _compat.file_part_bytes(result) == b""
    assert _compat.file_part_mime_type(result) == "text/plain"

  def test_convert_inline_data_part_with_video_metadata(self):
    """Test conversion of GenAI inline_data Part with video metadata to A2A Part."""
    # Arrange
    test_bytes = b"test video content"
    video_metadata = genai_types.VideoMetadata(fps=30.0)
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(data=test_bytes, mime_type="video/mp4"),
        video_metadata=video_metadata,
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_file_part(result)
    metadata = _compat.part_metadata(result)
    assert metadata
    assert _get_adk_metadata_key("video_metadata") in metadata

  def test_convert_inline_data_part_to_data_part(self):
    """Test conversion of GenAI inline_data Part to A2A DataPart."""
    # Arrange
    data = {"key": "value"}
    metadata = {"meta": "data"}
    # The embedded-payload shape and metadata channel differ between SDKs:
    #   0.3.x: the embedded JSON is the full DataPart ({"data", "metadata"}).
    #   1.x:   the embedded JSON is just the structured ``data`` dict and the
    #          metadata travels on the genai part's ``part_metadata``.
    if _compat.IS_A2A_V1:
      json_data = json.dumps(data).encode("utf-8")
      genai_part = genai_types.Part(
          inline_data=genai_types.Blob(
              data=A2A_DATA_PART_START_TAG + json_data + A2A_DATA_PART_END_TAG,
              mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
          ),
          part_metadata=metadata,
      )
    else:
      json_data = json.dumps({"data": data, "metadata": metadata}).encode(
          "utf-8"
      )
      genai_part = genai_types.Part(
          inline_data=genai_types.Blob(
              data=A2A_DATA_PART_START_TAG + json_data + A2A_DATA_PART_END_TAG,
              mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
          )
      )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    assert _compat.data_part_dict(result) == data
    assert _compat.part_metadata(result) == metadata

  def test_convert_function_call_part(self):
    """Test conversion of GenAI function_call Part to A2A Part."""
    # Arrange
    function_call = genai_types.FunctionCall(
        name="test_function", args={"param1": "value1", "param2": 42}
    )
    genai_part = genai_types.Part(function_call=function_call)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    expected_data = function_call.model_dump(by_alias=True, exclude_none=True)
    assert _normalize_numbers(
        _compat.data_part_dict(result)
    ) == _normalize_numbers(expected_data)
    assert (
        _compat.part_metadata(result)[
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        ]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )

  def test_convert_function_response_part(self):
    """Test conversion of GenAI function_response Part to A2A Part."""
    # Arrange
    function_response = genai_types.FunctionResponse(
        name="test_function", response={"result": "success", "data": [1, 2, 3]}
    )
    genai_part = genai_types.Part(function_response=function_response)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    expected_data = function_response.model_dump(
        by_alias=True, exclude_none=True
    )
    assert _normalize_numbers(
        _compat.data_part_dict(result)
    ) == _normalize_numbers(expected_data)
    assert (
        _compat.part_metadata(result)[
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        ]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
    )

  def test_convert_code_execution_result_part(self):
    """Test conversion of GenAI code_execution_result Part to A2A Part."""
    # Arrange
    code_execution_result = genai_types.CodeExecutionResult(
        outcome=genai_types.Outcome.OUTCOME_OK, output="Hello, World!"
    )
    genai_part = genai_types.Part(code_execution_result=code_execution_result)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    expected_data = code_execution_result.model_dump(
        by_alias=True, exclude_none=True
    )
    assert _normalize_numbers(
        _compat.data_part_dict(result)
    ) == _normalize_numbers(expected_data)
    assert (
        _compat.part_metadata(result)[
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        ]
        == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
    )

  def test_convert_executable_code_part(self):
    """Test conversion of GenAI executable_code Part to A2A Part."""
    # Arrange
    executable_code = genai_types.ExecutableCode(
        language=genai_types.Language.PYTHON, code="print('Hello, World!')"
    )
    genai_part = genai_types.Part(executable_code=executable_code)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    expected_data = executable_code.model_dump(by_alias=True, exclude_none=True)
    assert _normalize_numbers(
        _compat.data_part_dict(result)
    ) == _normalize_numbers(expected_data)
    assert (
        _compat.part_metadata(result)[
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        ]
        == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
    )

  def test_convert_unsupported_part(self):
    """Test handling of unsupported GenAI Part types."""
    # Arrange - Create a GenAI Part with no recognized fields
    genai_part = genai_types.Part()

    # Act
    with patch(
        "google.adk.a2a.converters.part_converter.logger"
    ) as mock_logger:
      result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is None
    mock_logger.warning.assert_called_once()


class TestRoundTripConversions:
  """Test cases for round-trip conversions to ensure consistency."""

  def test_text_part_round_trip(self):
    """Test round-trip conversion for text parts."""
    # Arrange
    original_text = "Hello, world!"
    a2a_part = _compat.make_text_part(original_text)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_text_part(result_a2a_part)
    assert _compat.part_text(result_a2a_part) == original_text

  def test_text_part_with_thought_round_trip(self):
    """Test round-trip conversion for text parts with thought."""
    # Arrange
    original_text = "Thinking..."
    genai_part = genai_types.Part(text=original_text, thought=True)

    # Act
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.text == original_text
    assert result_genai_part.thought

  def test_file_uri_round_trip(self):
    """Test round-trip conversion for file parts with URI."""
    # Arrange
    original_uri = "gs://bucket/file.txt"
    original_mime_type = "text/plain"
    a2a_part = _compat.make_file_part_with_uri(
        uri=original_uri, mime_type=original_mime_type
    )

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_file_part(result_a2a_part)
    assert _compat.file_part_uri(result_a2a_part) == original_uri
    assert _compat.file_part_mime_type(result_a2a_part) == original_mime_type

  def test_file_bytes_round_trip(self):
    """Test round-trip conversion for file parts with bytes."""
    # Arrange
    original_bytes = b"test file content for round trip"
    original_mime_type = "application/octet-stream"

    # Start with GenAI part (the more common starting point)
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=original_bytes, mime_type=original_mime_type
        )
    )

    # Act - Round trip: GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.inline_data is not None
    assert result_genai_part.inline_data.data == original_bytes
    assert result_genai_part.inline_data.mime_type == original_mime_type

  def test_function_call_round_trip(self):
    """Test round-trip conversion for function call parts."""
    # Arrange
    function_call = genai_types.FunctionCall(
        name="test_function", args={"param1": "value1", "param2": 42}
    )
    genai_part = genai_types.Part(function_call=function_call)

    # Act - Round trip: GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.function_call is not None
    assert result_genai_part.function_call.name == function_call.name
    assert result_genai_part.function_call.args == function_call.args

  def test_function_response_round_trip(self):
    """Test round-trip conversion for function response parts."""
    # Arrange
    function_response = genai_types.FunctionResponse(
        name="test_function", response={"result": "success", "data": [1, 2, 3]}
    )
    genai_part = genai_types.Part(function_response=function_response)

    # Act - Round trip: GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.function_response is not None
    assert result_genai_part.function_response.name == function_response.name
    assert (
        result_genai_part.function_response.response
        == function_response.response
    )

  def test_code_execution_result_round_trip(self):
    """Test round-trip conversion for code execution result parts."""
    # Arrange
    code_execution_result = genai_types.CodeExecutionResult(
        outcome=genai_types.Outcome.OUTCOME_OK, output="Hello, World!"
    )
    genai_part = genai_types.Part(code_execution_result=code_execution_result)

    # Act - Round trip: GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.code_execution_result is not None
    assert (
        result_genai_part.code_execution_result.outcome
        == code_execution_result.outcome
    )
    assert (
        result_genai_part.code_execution_result.output
        == code_execution_result.output
    )

  def test_executable_code_round_trip(self):
    """Test round-trip conversion for executable code parts."""
    # Arrange
    executable_code = genai_types.ExecutableCode(
        language=genai_types.Language.PYTHON, code="print('Hello, World!')"
    )
    genai_part = genai_types.Part(executable_code=executable_code)

    # Act - Round trip: GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(genai_part)
    result_genai_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result_genai_part is not None
    assert isinstance(result_genai_part, genai_types.Part)
    assert result_genai_part.executable_code is not None
    assert (
        result_genai_part.executable_code.language == executable_code.language
    )
    assert result_genai_part.executable_code.code == executable_code.code

  def test_data_part_round_trip(self):
    """Test round-trip conversion for data parts."""
    # Arrange
    data = {"key": "value"}
    metadata = {"meta": "data"}
    a2a_part = _compat.make_data_part(data=data, metadata=metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_data_part(result_a2a_part)
    assert _normalize_numbers(
        _compat.data_part_dict(result_a2a_part)
    ) == _normalize_numbers(data)
    assert _compat.part_metadata(result_a2a_part) == metadata

  def test_data_part_with_mime_type_metadata_round_trip(self):
    """Test round-trip conversion for data parts with 'mime_type' in metadata."""
    # Arrange
    data = {"content": "some data"}
    metadata = {"meta": "data", "mime_type": "application/json"}
    a2a_part = _compat.make_data_part(data=data, metadata=metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_data_part(result_a2a_part)
    assert _normalize_numbers(
        _compat.data_part_dict(result_a2a_part)
    ) == _normalize_numbers(data)
    # The 'mime_type' key in the metadata should be preserved as is
    assert _compat.part_metadata(result_a2a_part) == metadata

  def test_text_part_metadata_round_trip(self):
    """Test round-trip conversion for text parts with metadata."""
    # Arrange
    metadata = {"key1": "value1", "key2": "value2"}
    a2a_part = _compat.make_text_part("some text")
    _compat.set_part_metadata(a2a_part, metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_text_part(result_a2a_part)
    assert _compat.part_text(result_a2a_part) == "some text"
    assert _compat.part_metadata(result_a2a_part) == metadata

  def test_file_part_metadata_round_trip(self):
    """Test round-trip conversion for file parts with metadata."""
    # Arrange
    metadata = {"key1": "value1"}
    a2a_part = _compat.make_file_part_with_uri(
        uri="gs://bucket/file.txt",
        mime_type="text/plain",
        name="my_file.txt",
    )
    _compat.set_part_metadata(a2a_part, metadata)

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert _compat.is_file_part(result_a2a_part)
    assert _compat.file_part_uri(result_a2a_part) == "gs://bucket/file.txt"
    assert _compat.part_metadata(result_a2a_part) == metadata


class TestEdgeCases:
  """Test cases for edge cases and error conditions."""

  def test_empty_text_part(self):
    """Test conversion of empty text part."""
    # Arrange
    a2a_part = _compat.make_text_part("")

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.text == ""

  def test_genai_inline_data_with_mimetype_to_a2a(self):
    """Test conversion of GenAI inline_data with 'mimeType' in DataPart metadata to A2A.

    This tests if 'mimeType' in metadata of a DataPart wrapped in inline_data
    is correctly handled, ensuring the key casing is preserved.
    """
    # Arrange
    data = {"key": "value"}
    metadata = {"adk_type": "some_type", "mimeType": "image/png"}
    # The embedded-payload shape and metadata channel differ between SDKs.
    if _compat.IS_A2A_V1:
      json_data = json.dumps(data).encode("utf-8")
      genai_part = genai_types.Part(
          inline_data=genai_types.Blob(
              data=A2A_DATA_PART_START_TAG + json_data + A2A_DATA_PART_END_TAG,
              mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
          ),
          part_metadata=metadata,
      )
    else:
      json_data = json.dumps({"data": data, "metadata": metadata}).encode(
          "utf-8"
      )
      genai_part = genai_types.Part(
          inline_data=genai_types.Blob(
              data=A2A_DATA_PART_START_TAG + json_data + A2A_DATA_PART_END_TAG,
              mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
          )
      )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert _compat.is_data_part(result)
    assert _compat.data_part_dict(result) == data
    # The key casing should be preserved from the JSON
    assert _compat.part_metadata(result) == metadata

  def test_none_input_a2a_to_genai(self):
    """Test handling of None input for A2A to GenAI conversion."""
    # This test depends on how the function handles None input
    # If it should raise an exception, we test for that
    with pytest.raises(AttributeError):
      convert_a2a_part_to_genai_part(None)

  def test_none_input_genai_to_a2a(self):
    """Test handling of None input for GenAI to A2A conversion."""
    # This test depends on how the function handles None input
    # If it should raise an exception, we test for that
    with pytest.raises(AttributeError):
      convert_genai_part_to_a2a_part(None)


class TestNewConstants:
  """Test cases for new constants and functionality."""

  def test_new_constants_exist(self):
    """Test that new constants are defined."""
    assert (
        A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
        == "code_execution_result"
    )
    assert A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE == "executable_code"

  def test_convert_a2a_data_part_with_code_execution_result_metadata(self):
    """Test conversion of A2A DataPart with code execution result metadata."""
    # Arrange
    code_execution_result_data = {
        "outcome": "OUTCOME_OK",
        "output": "Hello, World!",
    }
    a2a_part = _compat.make_data_part(
        data=code_execution_result_data,
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT,
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    # Now it should convert back to a proper CodeExecutionResult
    assert result.code_execution_result is not None
    assert (
        result.code_execution_result.outcome == genai_types.Outcome.OUTCOME_OK
    )
    assert result.code_execution_result.output == "Hello, World!"

  def test_convert_a2a_data_part_with_executable_code_metadata(self):
    """Test conversion of A2A DataPart with executable code metadata."""
    # Arrange
    executable_code_data = {
        "language": "PYTHON",
        "code": "print('Hello, World!')",
    }
    a2a_part = _compat.make_data_part(
        data=executable_code_data,
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE,
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    # Now it should convert back to a proper ExecutableCode
    assert result.executable_code is not None
    assert result.executable_code.language == genai_types.Language.PYTHON
    assert result.executable_code.code == "print('Hello, World!')"


class TestThoughtSignaturePreservation:
  """Tests for thought_signature preservation in function call conversions."""

  def test_genai_function_call_with_thought_signature_to_a2a(self):
    """Test that thought_signature is preserved when converting GenAI to A2A."""
    # Arrange
    function_call = genai_types.FunctionCall(
        id="fc_gemini3",
        name="my_tool",
        args={"document": "test content"},
    )
    genai_part = genai_types.Part(
        function_call=function_call,
        thought_signature=b"gemini3_signature_bytes",
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert _compat.is_data_part(result)
    metadata = _compat.part_metadata(result)
    assert (
        metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )
    # thought_signature should be base64 encoded in metadata
    thought_sig_key = _get_adk_metadata_key("thought_signature")
    assert thought_sig_key in metadata
    assert (
        base64.b64decode(metadata[thought_sig_key])
        == b"gemini3_signature_bytes"
    )

  def test_genai_function_call_without_thought_signature_to_a2a(self):
    """Test function call without thought_signature doesn't add metadata key."""
    # Arrange
    function_call = genai_types.FunctionCall(
        id="fc_regular",
        name="regular_tool",
        args={},
    )
    genai_part = genai_types.Part(function_call=function_call)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert _compat.is_data_part(result)
    # thought_signature key should not be present
    thought_sig_key = _get_adk_metadata_key("thought_signature")
    assert thought_sig_key not in _compat.part_metadata(result)

  def test_a2a_function_call_with_thought_signature_to_genai(self):
    """Test that thought_signature is restored when converting A2A to GenAI."""
    # Arrange
    a2a_part = _compat.make_data_part(
        data={
            "id": "fc_gemini3",
            "name": "my_tool",
            "args": {"document": "test content"},
        },
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            _get_adk_metadata_key("thought_signature"): (
                base64.b64encode(b"restored_signature").decode("utf-8")
            ),
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.function_call is not None
    assert result.function_call.name == "my_tool"
    # thought_signature should be decoded back to bytes
    assert result.thought_signature == b"restored_signature"

  def test_a2a_function_call_without_thought_signature_to_genai(self):
    """Test function call without thought_signature returns None for it."""
    # Arrange
    a2a_part = _compat.make_data_part(
        data={
            "id": "fc_regular",
            "name": "regular_tool",
            "args": {},
        },
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.function_call is not None
    assert result.function_call.name == "regular_tool"
    # thought_signature should be None
    assert result.thought_signature is None

  def test_function_call_with_thought_signature_round_trip(self):
    """Test thought_signature is preserved in GenAI -> A2A -> GenAI round trip."""
    # Arrange
    original_signature = b"round_trip_signature_test"
    function_call = genai_types.FunctionCall(
        id="fc_round_trip",
        name="round_trip_tool",
        args={"key": "value"},
    )
    original_part = genai_types.Part(
        function_call=function_call,
        thought_signature=original_signature,
    )

    # Act - Convert GenAI -> A2A -> GenAI
    a2a_part = convert_genai_part_to_a2a_part(original_part)
    restored_part = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert restored_part is not None
    assert restored_part.function_call is not None
    assert restored_part.function_call.name == "round_trip_tool"
    assert restored_part.thought_signature == original_signature

  @pytest.mark.skipif(
      _compat.IS_A2A_V1,
      reason=(
          "0.3-only: 1.x metadata is a proto Struct that cannot hold raw bytes"
      ),
  )
  def test_a2a_function_call_with_bytes_thought_signature_to_genai(self):
    """Test that bytes thought_signature is used directly without decoding."""
    # Arrange - metadata contains raw bytes (not base64 encoded)
    a2a_part = _compat.make_data_part(
        data={
            "id": "fc_bytes",
            "name": "bytes_tool",
            "args": {},
        },
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            _get_adk_metadata_key("thought_signature"): b"raw_bytes_signature",
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.function_call is not None
    # bytes should be used directly
    assert result.thought_signature == b"raw_bytes_signature"

  def test_a2a_function_call_with_invalid_base64_thought_signature(self):
    """Test that invalid base64 thought_signature logs warning and returns None."""
    # Arrange - metadata contains invalid base64 string
    a2a_part = _compat.make_data_part(
        data={
            "id": "fc_invalid",
            "name": "invalid_sig_tool",
            "args": {},
        },
        metadata={
            _get_adk_metadata_key(
                A2A_DATA_PART_METADATA_TYPE_KEY
            ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            _get_adk_metadata_key("thought_signature"): "not_valid_base64!!!",
        },
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.function_call is not None
    assert result.function_call.name == "invalid_sig_tool"
    # thought_signature should be None due to decode failure
    assert result.thought_signature is None


class TestBytesSerialization:
  """Tests that raw bytes serialize as base64 through the A2A converters."""

  def _function_response_with_bytes(self) -> genai_types.Part:
    screenshot = b"\x89PNG\r\n\x1a\n_FAKE_SCREENSHOT_" + bytes(range(16))
    function_response = genai_types.FunctionResponse(
        name="computer_use",
        response={
            "screenshot": {"inline_data": {"data": screenshot}},
            "status": "ok",
        },
    )
    return genai_types.Part(function_response=function_response)

  def _assert_bytes_serialized_as_base64_str(self, a2a_part):
    assert a2a_part is not None
    assert _compat.is_data_part(a2a_part)
    # Raw bytes must be serialized to a base64 str, not kept as bytes.
    data_dict = _compat.data_part_dict(a2a_part)
    serialized = data_dict["response"]["screenshot"]["inline_data"]["data"]
    assert isinstance(serialized, str)

  @pytest.mark.skipif(_compat.IS_A2A_V1, reason="0.3-only proto_utils.ToProto")
  def test_function_response_with_bytes_serializes_to_proto_struct_v03(self):
    """0.3: the A2A DataPart serializes to a proto Struct without raising."""
    from a2a.utils import proto_utils

    a2a_part = convert_genai_part_to_a2a_part(
        self._function_response_with_bytes()
    )
    self._assert_bytes_serialized_as_base64_str(a2a_part)

    proto_part = proto_utils.ToProto.part(a2a_part)
    assert proto_part is not None

  @pytest.mark.skipif(
      not _compat.IS_A2A_V1, reason="1.x-only proto Part / Struct"
  )
  def test_function_response_with_bytes_serializes_to_proto_struct_v1x(self):
    """1.x: conversion builds the proto Struct in place (via ParseDict)."""
    from google.protobuf import struct_pb2

    a2a_part = convert_genai_part_to_a2a_part(
        self._function_response_with_bytes()
    )
    self._assert_bytes_serialized_as_base64_str(a2a_part)

    # The proto Struct build already happened during conversion; assert it.
    assert a2a_part.WhichOneof("content") == "data"
    assert isinstance(a2a_part.data, struct_pb2.Value)
    assert a2a_part.data.HasField("struct_value")

  def test_function_response_with_bytes_round_trip(self):
    """genai -> a2a -> genai restores the original bytes losslessly."""
    original = self._function_response_with_bytes()
    a2a_part = convert_genai_part_to_a2a_part(original)
    result = convert_a2a_part_to_genai_part(a2a_part)

    assert result is not None
    assert result.function_response is not None
    restored = result.function_response.response["screenshot"]["inline_data"][
        "data"
    ]
    expected = original.function_response.response["screenshot"]["inline_data"][
        "data"
    ]
    if isinstance(restored, str):
      restored = base64.b64decode(restored)
    assert restored == expected
