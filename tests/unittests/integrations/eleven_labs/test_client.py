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

"""Tests for google.adk.integrations.eleven_labs._client."""

from __future__ import annotations

import builtins
import os
from unittest import mock

from google.adk.integrations.eleven_labs._client import API_KEY_ENV
from google.adk.integrations.eleven_labs._client import load_elevenlabs
from google.adk.integrations.eleven_labs._client import resolve_api_key
import pytest


def test_resolve_api_key_explicit():
  assert resolve_api_key("my-explicit-key") == "my-explicit-key"


def test_resolve_api_key_env_var():
  with mock.patch.dict(os.environ, {API_KEY_ENV: "env-key"}):
    assert resolve_api_key() == "env-key"


def test_resolve_api_key_missing_raises():
  with mock.patch.dict(os.environ, {}, clear=True):
    with pytest.raises(
        ValueError, match="ELEVENLABS_API_KEY environment variable is not set"
    ):
      resolve_api_key()


def test_load_elevenlabs_success():
  fake_module = mock.MagicMock()
  with mock.patch.dict("sys.modules", {"elevenlabs": fake_module}):
    assert load_elevenlabs() is fake_module


def test_load_elevenlabs_missing_raises():
  real_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    if name == "elevenlabs":
      raise ImportError("No module named elevenlabs")
    return real_import(name, *args, **kwargs)

  with mock.patch("builtins.__import__", side_effect=mock_import):
    with pytest.raises(ImportError, match="requires the ElevenLabs SDK"):
      load_elevenlabs()
