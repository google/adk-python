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

import json
from unittest.mock import Mock
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
from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.genai import types as genai_types
import pytest


class TestConvertA2aPartToGenaiPart:
  """Test cases for convert_a2a_part_to_genai_part function."""

  def test_convert_text_part(self):
    """Test conversion of A2A TextPart to GenAI Part."""
    # Arrange
    a2a_part = a2a_types.Part(root=a2a_types.TextPart(text="Hello, world!"))

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.text == "Hello, world!"

  def test_convert_file_part_with_uri(self):
    """Test conversion of A2A FilePart with URI to GenAI Part."""
    # Arrange
    a2a_part = a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithUri(
                uri="gs://bucket/file.txt", mime_type="text/plain"
            )
        )
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.file_data is not None
    assert result.file_data.file_uri == "gs://bucket/file.txt"
    assert result.file_data.mime_type == "text/plain"

  def test_convert_file_part_with_bytes(self):
    """Test conversion of A2A FilePart with bytes to GenAI Part."""
    # Arrange
    test_bytes = b"test file content"
    # A2A FileWithBytes expects base64-encoded string
    import base64

    base64_encoded = base64.b64encode(test_bytes).decode("utf-8")
    a2a_part = a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithBytes(
                bytes=base64_encoded, mime_type="text/plain"
            )
        )
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

  def test_convert_data_part_function_call(self):
    """Test conversion of A2A DataPart with function call metadata."""
    # Arrange
    function_call_data = {
        "name": "test_function",
        "args": {"param1": "value1", "param2": 42},
    }
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(
            data=function_call_data,
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
                "adk_type": A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
            },
        )
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
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(
            data=function_response_data,
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
                "adk_type": A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
            },
        )
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
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(data=data, metadata=metadata)
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.inline_data is not None
    assert result.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
    assert result.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
    assert result.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    converted_data_part = a2a_types.DataPart.model_validate_json(
        result.inline_data.data[
            len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
        ]
    )
    assert converted_data_part.data == data
    assert converted_data_part.metadata == metadata

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
    assert isinstance(result.root, a2a_types.TextPart)
    assert result.root.text == "Hello, world!"

  def test_convert_text_part_with_thought(self):
    """Test conversion of GenAI text Part with thought to A2A Part."""
    # Arrange - thought is a boolean field in genai_types.Part
    genai_part = genai_types.Part(text="Hello, world!", thought=True)

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert isinstance(result.root, a2a_types.TextPart)
    assert result.root.text == "Hello, world!"
    assert result.root.metadata is not None
    assert result.root.metadata[_get_adk_metadata_key("thought")] == True

  def test_convert_file_data_part(self):
    """Test conversion of GenAI file_data Part to A2A Part."""
    # Arrange
    genai_part = genai_types.Part(
        file_data=genai_types.FileData(
            file_uri="gs://bucket/file.txt", mime_type="text/plain"
        )
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert isinstance(result.root, a2a_types.FilePart)
    assert isinstance(result.root.file, a2a_types.FileWithUri)
    assert result.root.file.uri == "gs://bucket/file.txt"
    assert result.root.file.mime_type == "text/plain"

  def test_convert_inline_data_part(self):
    """Test conversion of GenAI inline_data Part to A2A Part."""
    # Arrange
    test_bytes = b"test file content"
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(data=test_bytes, mime_type="text/plain")
    )

    # Act
    result = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result is not None
    assert isinstance(result, a2a_types.Part)
    assert isinstance(result.root, a2a_types.FilePart)
    assert isinstance(result.root.file, a2a_types.FileWithBytes)
    # A2A FileWithBytes now stores base64-encoded bytes to ensure round-trip compatibility
    import base64

    expected_base64 = base64.b64encode(test_bytes).decode("utf-8")
    assert result.root.file.bytes == expected_base64
    assert result.root.file.mime_type == "text/plain"

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
    assert isinstance(result.root, a2a_types.FilePart)
    assert isinstance(result.root.file, a2a_types.FileWithBytes)
    assert result.root.metadata is not None
    assert _get_adk_metadata_key("video_metadata") in result.root.metadata

  def test_convert_inline_data_part_to_data_part(self):
    """Test conversion of GenAI inline_data Part to A2A DataPart."""
    # Arrange
    data = {"key": "value"}
    metadata = {"meta": "data"}
    a2a_part_to_convert = a2a_types.DataPart(data=data, metadata=metadata)
    json_data = a2a_part_to_convert.model_dump_json(
        by_alias=True, exclude_none=True
    ).encode("utf-8")
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
    assert isinstance(result.root, a2a_types.DataPart)
    assert result.root.data == data
    assert result.root.metadata == metadata

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
    assert isinstance(result.root, a2a_types.DataPart)
    expected_data = function_call.model_dump(by_alias=True, exclude_none=True)
    assert result.root.data == expected_data
    assert (
        result.root.metadata[
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
    assert isinstance(result.root, a2a_types.DataPart)
    expected_data = function_response.model_dump(
        by_alias=True, exclude_none=True
    )
    assert result.root.data == expected_data
    assert (
        result.root.metadata[
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
    assert isinstance(result.root, a2a_types.DataPart)
    expected_data = code_execution_result.model_dump(
        by_alias=True, exclude_none=True
    )
    assert result.root.data == expected_data
    assert (
        result.root.metadata[
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
    assert isinstance(result.root, a2a_types.DataPart)
    expected_data = executable_code.model_dump(by_alias=True, exclude_none=True)
    assert result.root.data == expected_data
    assert (
        result.root.metadata[
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
    a2a_part = a2a_types.Part(root=a2a_types.TextPart(text=original_text))

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert isinstance(result_a2a_part.root, a2a_types.TextPart)
    assert result_a2a_part.root.text == original_text

  def test_file_uri_round_trip(self):
    """Test round-trip conversion for file parts with URI."""
    # Arrange
    original_uri = "gs://bucket/file.txt"
    original_mime_type = "text/plain"
    a2a_part = a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithUri(
                uri=original_uri, mime_type=original_mime_type
            )
        )
    )

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert isinstance(result_a2a_part.root, a2a_types.FilePart)
    assert isinstance(result_a2a_part.root.file, a2a_types.FileWithUri)
    assert result_a2a_part.root.file.uri == original_uri
    assert result_a2a_part.root.file.mime_type == original_mime_type

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
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(data=data, metadata=metadata)
    )

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert isinstance(result_a2a_part.root, a2a_types.DataPart)
    assert result_a2a_part.root.data == data
    assert result_a2a_part.root.metadata == metadata

  def test_data_part_with_mime_type_metadata_round_trip(self):
    """Test round-trip conversion for data parts with 'mime_type' in metadata."""
    # Arrange
    data = {"content": "some data"}
    metadata = {"meta": "data", "mime_type": "application/json"}
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(data=data, metadata=metadata)
    )

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.Part)
    assert isinstance(result_a2a_part.root, a2a_types.DataPart)
    assert result_a2a_part.root.data == data
    # The 'mime_type' key in the metadata should be preserved as is
    assert result_a2a_part.root.metadata == metadata


class TestEdgeCases:
  """Test cases for edge cases and error conditions."""

  def test_empty_text_part(self):
    """Test conversion of empty text part."""
    # Arrange
    a2a_part = a2a_types.Part(root=a2a_types.TextPart(text=""))

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
    a2a_part_inner = a2a_types.DataPart(data=data, metadata=metadata)
    json_data = a2a_part_inner.model_dump_json(
        by_alias=True, exclude_none=True
    ).encode("utf-8")
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
    assert isinstance(result.root, a2a_types.DataPart)
    assert result.root.data == data
    # The key casing should be preserved from the JSON
    assert result.root.metadata == metadata

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
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(
            data=code_execution_result_data,
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT,
            },
        )
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
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(
            data=executable_code_data,
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE,
            },
        )
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
