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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.models.keras_llm import KerasLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


class TestKerasLlm:
  """Test cases for KerasLlm."""

  @pytest.fixture
  def keras_llm_instance(self):
    """Create a KerasLlm instance for testing."""
    return KerasLlm(model="gpt2_base_en", max_length=50)

  def test_init(self, keras_llm_instance):
    """Test KerasLlm initialization."""
    assert keras_llm_instance.model == "gpt2_base_en"
    assert keras_llm_instance._additional_args["max_length"] == 50
    assert keras_llm_instance._keras_model is None
    assert keras_llm_instance._preprocessor is None

  def test_supported_models(self):
    """Test supported_models method."""
    supported = KerasLlm.supported_models()
    assert "keras/.*" in supported
    assert "gpt2_.*_en" in supported
    assert "opt_.*_en" in supported

  def test_load_model_import_error(self, keras_llm_instance):
    """Test load_model when keras_hub is not installed."""
    with patch(
        "builtins.__import__",
        side_effect=ImportError("No module named 'keras_hub'"),
    ):
      with pytest.raises(ImportError, match="KerasHub is not installed"):
        keras_llm_instance._load_model()

  @patch("keras_hub.models.CausalLM.from_preset")
  def test_load_model_success(self, mock_from_preset, keras_llm_instance):
    """Test successful model loading."""
    mock_model = Mock()
    mock_model.compile = Mock()
    mock_from_preset.return_value = mock_model

    keras_llm_instance._load_model()

    mock_from_preset.assert_called_once_with("gpt2_base_en", load_weights=True)
    # Check that compile was called with a sampler instance (not a string)
    mock_model.compile.assert_called_once()
    call_args = mock_model.compile.call_args
    assert call_args is not None
    assert "sampler" in call_args.kwargs
    assert hasattr(call_args.kwargs["sampler"], "__class__")

  @patch("keras_hub.models.CausalLM.from_preset")
  def test_load_model_with_sampler(self, mock_from_preset):
    """Test model loading with different samplers."""
    mock_model = Mock()
    mock_model.compile = Mock()
    mock_from_preset.return_value = mock_model

    # Test top_k sampler
    keras_llm = KerasLlm(model="gpt2_base_en", sampler="top_k", top_k=30)
    keras_llm._load_model()
    mock_model.compile.assert_called_once()
    call_args = mock_model.compile.call_args
    assert call_args is not None
    assert "sampler" in call_args.kwargs
    sampler = call_args.kwargs["sampler"]
    assert hasattr(sampler, "__class__")
    assert "TopKSampler" in sampler.__class__.__name__

    # Test top_p sampler
    mock_model.compile.reset_mock()
    keras_llm = KerasLlm(model="gpt2_base_en", sampler="top_p", top_p=0.8)
    keras_llm._load_model()
    mock_model.compile.assert_called_once()
    call_args = mock_model.compile.call_args
    assert call_args is not None
    assert "sampler" in call_args.kwargs
    sampler = call_args.kwargs["sampler"]
    assert hasattr(sampler, "__class__")
    assert "TopPSampler" in sampler.__class__.__name__

    # Test beam sampler
    mock_model.compile.reset_mock()
    keras_llm = KerasLlm(model="gpt2_base_en", sampler="beam", num_beams=5)
    keras_llm._load_model()
    mock_model.compile.assert_called_once()
    call_args = mock_model.compile.call_args
    assert call_args is not None
    assert "sampler" in call_args.kwargs
    sampler = call_args.kwargs["sampler"]
    assert hasattr(sampler, "__class__")
    assert "BeamSampler" in sampler.__class__.__name__

  def test_flatten_conversation_to_prompt(self, keras_llm_instance):
    """Test conversation flattening to prompt."""
    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="system",
                parts=[
                    types.Part.from_text(text="You are a helpful assistant.")
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello, how are you?")],
            ),
            types.Content(
                role="assistant",
                parts=[types.Part.from_text(text="I'm doing well, thank you!")],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="What's the weather like?")],
            ),
        ]
    )

    prompt = keras_llm_instance._flatten_conversation_to_prompt(llm_request)
    expected = (
        "System: You are a helpful assistant.\n"
        "User: Hello, how are you?\n"
        "Assistant: I'm doing well, thank you!\n"
        "User: What's the weather like?\n"
        "Assistant:"
    )
    assert prompt == expected

  def test_flatten_conversation_empty(self, keras_llm_instance):
    """Test conversation flattening with empty request."""
    llm_request = LlmRequest(contents=[])
    prompt = keras_llm_instance._flatten_conversation_to_prompt(llm_request)
    assert prompt == "Assistant:"

  @pytest.mark.asyncio
  @patch("asyncio.to_thread")
  @patch("keras_hub.models.CausalLM.from_preset")
  async def test_generate_content_async_success(
      self, mock_from_preset, mock_to_thread, keras_llm_instance
  ):
    """Test successful content generation."""
    # Setup mocks
    mock_model = Mock()
    mock_model.compile = Mock()
    mock_from_preset.return_value = mock_model
    mock_to_thread.return_value = "This is a test response."

    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="Hello")]
            )
        ]
    )

    # Test generation
    responses = []
    async for response in keras_llm_instance.generate_content_async(
        llm_request
    ):
      responses.append(response)

    assert len(responses) == 1
    assert responses[0].content.role == "assistant"
    assert responses[0].content.parts[0].text == "This is a test response."

    # Verify model was loaded
    mock_from_preset.assert_called_once()
    mock_to_thread.assert_called_once()

  @pytest.mark.asyncio
  @patch("asyncio.to_thread")
  @patch("keras_hub.models.CausalLM.from_preset")
  async def test_generate_content_async_error(
      self, mock_from_preset, mock_to_thread, keras_llm_instance
  ):
    """Test content generation with error."""
    # Setup mocks
    mock_model = Mock()
    mock_model.compile = Mock()
    mock_from_preset.return_value = mock_model
    mock_to_thread.side_effect = RuntimeError("Model generation failed")

    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="Hello")]
            )
        ]
    )

    # Test error handling
    with pytest.raises(
        RuntimeError, match="Failed to generate text with KerasHub model"
    ):
      async for _ in keras_llm_instance.generate_content_async(llm_request):
        pass

  @pytest.mark.asyncio
  async def test_generate_content_async_maybe_append_user_content(
      self, keras_llm_instance
  ):
    """Test that _maybe_append_user_content is called."""
    llm_request = LlmRequest(contents=[])

    with patch.object(
        keras_llm_instance, "_maybe_append_user_content"
    ) as mock_append:
      with patch.object(keras_llm_instance, "_load_model"):
        with patch.object(
            keras_llm_instance,
            "_flatten_conversation_to_prompt",
            return_value="test",
        ):
          with patch("asyncio.to_thread", return_value="response"):
            async for _ in keras_llm_instance.generate_content_async(
                llm_request
            ):
              pass

            mock_append.assert_called_once_with(llm_request)
