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

import json
import sys
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a import types as a2a_types
  from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
  from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
  from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
  from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
  from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
  from google.genai import types as genai_types
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    a2a_types = DummyTypes()
    genai_types = DummyTypes()
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL = "function_call"
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE = "function_response"
    A2A_DATA_PART_METADATA_TYPE_KEY = "type"
    convert_a2a_part_to_genai_part = lambda x: None
    convert_genai_part_to_a2a_part = lambda x: None
  else:
    raise e


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
                uri="gs://bucket/file.txt", mimeType="text/plain"
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
    # Note: A2A FileWithBytes converts bytes to string automatically
    a2a_part = a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithBytes(
                bytes=test_bytes, mimeType="text/plain"
            )
        )
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.inline_data is not None
    # Source code now properly converts A2A string back to bytes for GenAI Blob
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
                A2A_DATA_PART_METADATA_TYPE_KEY: (
                    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
                ),
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
                A2A_DATA_PART_METADATA_TYPE_KEY: (
                    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
                ),
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

  def test_convert_data_part_without_special_metadata(self):
    """Test conversion of A2A DataPart without special metadata to text."""
    # Arrange
    data = {"key": "value", "number": 123}
    a2a_part = a2a_types.Part(
        root=a2a_types.DataPart(data=data, metadata={"other": "metadata"})
    )

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.text == json.dumps(data)

  def test_convert_data_part_no_metadata(self):
    """Test conversion of A2A DataPart with no metadata to text."""
    # Arrange
    data = {"key": "value", "array": [1, 2, 3]}
    a2a_part = a2a_types.Part(root=a2a_types.DataPart(data=data))

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert isinstance(result, genai_types.Part)
    assert result.text == json.dumps(data)

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
    assert isinstance(result, a2a_types.TextPart)
    assert result.text == "Hello, world!"

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
    assert isinstance(result, a2a_types.FilePart)
    assert isinstance(result.file, a2a_types.FileWithUri)
    assert result.file.uri == "gs://bucket/file.txt"
    assert result.file.mimeType == "text/plain"

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
    # A2A FileWithBytes stores bytes as strings
    assert result.root.file.bytes == test_bytes.decode("utf-8")
    assert result.root.file.mimeType == "text/plain"

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
        result.root.metadata[A2A_DATA_PART_METADATA_TYPE_KEY]
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
        result.root.metadata[A2A_DATA_PART_METADATA_TYPE_KEY]
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
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
    assert isinstance(result_a2a_part, a2a_types.TextPart)
    assert result_a2a_part.text == original_text

  def test_file_uri_round_trip(self):
    """Test round-trip conversion for file parts with URI."""
    # Arrange
    original_uri = "gs://bucket/file.txt"
    original_mime_type = "text/plain"
    a2a_part = a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithUri(
                uri=original_uri, mimeType=original_mime_type
            )
        )
    )

    # Act
    genai_part = convert_a2a_part_to_genai_part(a2a_part)
    result_a2a_part = convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert result_a2a_part is not None
    assert isinstance(result_a2a_part, a2a_types.FilePart)
    assert isinstance(result_a2a_part.file, a2a_types.FileWithUri)
    assert result_a2a_part.file.uri == original_uri
    assert result_a2a_part.file.mimeType == original_mime_type


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

  def test_data_part_with_complex_data(self):
    """Test conversion of DataPart with complex nested data."""
    # Arrange
    complex_data = {
        "nested": {
            "array": [1, 2, {"inner": "value"}],
            "boolean": True,
            "null_value": None,
        },
        "unicode": "Hello 世界 🌍",
    }
    a2a_part = a2a_types.Part(root=a2a_types.DataPart(data=complex_data))

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.text == json.dumps(complex_data)

  def test_data_part_with_empty_metadata(self):
    """Test conversion of DataPart with empty metadata dict."""
    # Arrange
    data = {"key": "value"}
    a2a_part = a2a_types.Part(root=a2a_types.DataPart(data=data, metadata={}))

    # Act
    result = convert_a2a_part_to_genai_part(a2a_part)

    # Assert
    assert result is not None
    assert result.text == json.dumps(data)
