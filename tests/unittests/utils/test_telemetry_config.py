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

"""Tests for telemetry_config.py."""

import json
import pathlib
import unittest.mock

from google.adk.utils import _telemetry_config as telemetry_config
import pytest


def test_get_user_config_path():
  path = telemetry_config.get_user_config_path()
  assert isinstance(path, pathlib.Path)
  assert path.name == "config.json"
  assert path.parent.name == ".adk"


def test_read_telemetry_consent_not_exists(tmp_path):
  config_file = tmp_path / "config.json"
  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    assert telemetry_config.read_telemetry_consent() is None


def test_read_telemetry_consent_exists_true(tmp_path):
  config_file = tmp_path / "config.json"
  config_file.parent.mkdir(parents=True, exist_ok=True)
  with open(config_file, "w", encoding="utf-8") as f:
    json.dump({"telemetry": True}, f)

  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    assert telemetry_config.read_telemetry_consent() is True


def test_read_telemetry_consent_exists_false(tmp_path):
  config_file = tmp_path / "config.json"
  config_file.parent.mkdir(parents=True, exist_ok=True)
  with open(config_file, "w", encoding="utf-8") as f:
    json.dump({"telemetry": False}, f)

  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    assert telemetry_config.read_telemetry_consent() is False


def test_read_telemetry_consent_invalid_json(tmp_path):
  config_file = tmp_path / "config.json"
  config_file.parent.mkdir(parents=True, exist_ok=True)
  with open(config_file, "w", encoding="utf-8") as f:
    f.write("not raw json data")

  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    assert telemetry_config.read_telemetry_consent() is None


def test_write_telemetry_consent(tmp_path):
  config_file = tmp_path / "config.json"
  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    telemetry_config.write_telemetry_consent(True)
    assert telemetry_config.read_telemetry_consent() is True

    # Overwrite
    telemetry_config.write_telemetry_consent(False)
    assert telemetry_config.read_telemetry_consent() is False

    with open(config_file, "r", encoding="utf-8") as f:
      data = json.load(f)
      assert data == {"telemetry": False}


def test_write_telemetry_consent_raises_on_error(tmp_path):
  config_file = tmp_path / "config.json"
  with unittest.mock.patch.object(
      telemetry_config, "get_user_config_path", return_value=config_file
  ):
    with unittest.mock.patch.object(
        pathlib.Path, "mkdir", side_effect=OSError("Permission denied")
    ):
      with pytest.raises(OSError):
        telemetry_config.write_telemetry_consent(True)
