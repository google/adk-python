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

from google.adk.agents.run_config import RunConfig
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
