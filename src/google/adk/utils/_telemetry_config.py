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

"""Telemetry user consent configuration utilities."""

from __future__ import annotations

import json
import logging
import pathlib
import typing

logger = logging.getLogger("google_adk." + __name__)


def get_user_config_path() -> pathlib.Path:
  """Returns the path to the ADK global config file."""
  return pathlib.Path.home() / ".adk" / "config.json"


def read_telemetry_consent() -> typing.Optional[bool]:
  """Reads the telemetry consent status from local config (config.json).

  Returns:
    True if opted-in, False if opted-out, and None if no explicit
    preference has been recorded yet or if there is an error reading
    the config.
  """
  path = get_user_config_path()
  if not path.exists():
    return None
  try:
    with open(path, "r", encoding="utf-8") as f:
      config = json.load(f)
      val = config.get("telemetry", None)
      if isinstance(val, bool):
        return val
      return None
  except Exception as e:
    logger.warning("Failed to read telemetry config from %s: %s", path, e)
    return None


def write_telemetry_consent(enabled: bool) -> None:
  """Writes the telemetry consent status to local config (config.json)."""
  path = get_user_config_path()
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if path.exists():
      try:
        with open(path, "r", encoding="utf-8") as f:
          config = json.load(f)
          if not isinstance(config, dict):
            config = {}
      except Exception:
        # If config parsing fails, start with an empty dictionary
        config = {}
    config["telemetry"] = enabled
    with open(path, "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)
      f.write("\n")
  except Exception as e:
    logger.error("Failed to write telemetry config to %s: %s", path, e)
    raise
