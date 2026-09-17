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

import sys
from unittest.mock import ANY
from unittest.mock import patch
import warnings

from google.adk.agents.run_config import RunConfig
from google.genai import types
import pytest


def test_validate_max_llm_calls_valid():
  value = RunConfig.validate_max_llm_calls(100)
  assert value == 100


def test_validate_max_llm_calls_negative():
  with patch("google.adk.agents.run_config.logger.warning") as mock_warning:
    value = RunConfig.validate_max_llm_calls(-1)
    mock_warning.assert_called_once_with(ANY)
    assert value == -1


def test_validate_max_llm_calls_warns_on_zero():
  with patch("google.adk.agents.run_config.logger.warning") as mock_warning:
    value = RunConfig.validate_max_llm_calls(0)
    mock_warning.assert_called_once_with(ANY)
    assert value == 0


def test_validate_max_llm_calls_too_large():
  with pytest.raises(
      ValueError, match=f"max_llm_calls should be less than {sys.maxsize}."
  ):
    RunConfig.validate_max_llm_calls(sys.maxsize)


def test_audio_transcription_configs_are_not_shared_between_instances():
  config1 = RunConfig()
  config2 = RunConfig()

  # Validate output_audio_transcription
  assert config1.output_audio_transcription is not None
  assert config2.output_audio_transcription is not None
  assert (
      config1.output_audio_transcription
      is not config2.output_audio_transcription
  )

  # Validate input_audio_transcription
  assert config1.input_audio_transcription is not None
  assert config2.input_audio_transcription is not None
  assert (
      config1.input_audio_transcription is not config2.input_audio_transcription
  )


def test_response_modalities_accepts_enum():
  config = RunConfig(response_modalities=[types.Modality.AUDIO])
  assert config.response_modalities == [types.Modality.AUDIO]
  assert isinstance(config.response_modalities[0], types.Modality)


def test_response_modalities_coerces_string_to_enum():
  config = RunConfig(response_modalities=["AUDIO"])
  assert config.response_modalities == [types.Modality.AUDIO]
  assert isinstance(config.response_modalities[0], types.Modality)


def test_response_modalities_coerces_lowercase_string_to_enum():
  config = RunConfig(response_modalities=["audio"])
  assert config.response_modalities == [types.Modality.AUDIO]
  assert isinstance(config.response_modalities[0], types.Modality)


def test_response_modalities_serialization_no_warning():
  config = RunConfig(response_modalities=[types.Modality.AUDIO])
  live_config = types.LiveConnectConfig()
  live_config.response_modalities = config.response_modalities
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    live_config.model_dump()
    pydantic_warnings = [
        x for x in w if "PydanticSerializationUnexpectedValue" in str(x.message)
    ]
    assert len(pydantic_warnings) == 0


def test_avatar_config_initialization():
  custom_avatar = types.CustomizedAvatar(
      image_mime_type="image/jpeg", image_data=b"image_bytes"
  )
  avatar_config = types.AvatarConfig(
      audio_bitrate_bps=128000,
      video_bitrate_bps=1000000,
      customized_avatar=custom_avatar,
  )
  run_config = RunConfig(avatar_config=avatar_config)

  assert run_config.avatar_config == avatar_config
  assert run_config.avatar_config.customized_avatar == custom_avatar
  assert (
      run_config.avatar_config.customized_avatar.image_mime_type == "image/jpeg"
  )
  assert run_config.avatar_config.customized_avatar.image_data == b"image_bytes"


def test_avatar_config_with_name():
  avatar_config = types.AvatarConfig(
      audio_bitrate_bps=128000,
      video_bitrate_bps=1000000,
      avatar_name="test_avatar",
  )
  run_config = RunConfig(avatar_config=avatar_config)

  assert run_config.avatar_config == avatar_config
  assert run_config.avatar_config.avatar_name == "test_avatar"
  assert run_config.avatar_config.customized_avatar is None


def test_model_input_context_accepts_transient_contents():
  context_content = types.UserContent("Relevant context for this turn")

  run_config = RunConfig(model_input_context=[context_content])

  assert run_config.model_input_context == [context_content]


def _deprecation_messages(records) -> list[str]:
  return [
      str(record.message)
      for record in records
      if issubclass(record.category, DeprecationWarning)
  ]


def test_save_live_audio_true_turns_on_save_live_blob():
  """The deprecated flag must keep working by forwarding to its replacement."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = RunConfig(save_live_audio=True)

  assert config.save_live_blob is True
  assert any(
      "`save_live_audio` config is deprecated" in message
      for message in _deprecation_messages(caught)
  )


def test_save_live_audio_false_leaves_save_live_blob_off():
  """Opting out of the deprecated flag must not opt in to the new one."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = RunConfig(save_live_audio=False)

  assert config.save_live_blob is False
  assert any(
      "`save_live_audio` config is deprecated" in message
      for message in _deprecation_messages(caught)
  )


def test_save_live_audio_overrides_explicit_save_live_blob_false():
  """When both are given, the caller asked for blobs to be saved."""
  config = RunConfig(save_live_audio=True, save_live_blob=False)

  assert config.save_live_blob is True


def test_no_deprecation_warning_when_save_live_audio_is_not_passed():
  """Callers who never touched the deprecated field must not be warned."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = RunConfig(save_live_blob=True)

  assert config.save_live_blob is True
  assert not [
      message
      for message in _deprecation_messages(caught)
      if "`save_live_audio` config is deprecated" in message
  ]


def test_max_llm_calls_default_value(monkeypatch):
  monkeypatch.delenv("ADK_MAX_LLM_CALLS", raising=False)
  config = RunConfig()
  assert config.max_llm_calls == 500


def test_max_llm_calls_env_var_override(monkeypatch):
  monkeypatch.setenv("ADK_MAX_LLM_CALLS", "100")
  config = RunConfig()
  assert config.max_llm_calls == 100


def test_max_llm_calls_explicit_value_overrides_env_var(monkeypatch):
  monkeypatch.setenv("ADK_MAX_LLM_CALLS", "100")
  config = RunConfig(max_llm_calls=200)
  assert config.max_llm_calls == 200


def test_max_llm_calls_invalid_env_var_warning(monkeypatch):
  monkeypatch.setenv("ADK_MAX_LLM_CALLS", "invalid")
  with patch("google.adk.agents.run_config.logger.warning") as mock_warning:
    config = RunConfig()
    assert config.max_llm_calls == 500
    mock_warning.assert_called_once()
    assert "Invalid value for ADK_MAX_LLM_CALLS" in mock_warning.call_args[0][0]
