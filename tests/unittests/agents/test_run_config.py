import logging
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
