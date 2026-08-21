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

"""Config shared by both entry points."""

from __future__ import annotations

import os

# Also the LiveKit `agent_name` the worker registers and the client dispatches.
APP_NAME = "roll_dice"


def require_env(name: str) -> str:
  """Returns an environment variable, or explains which one is missing."""
  try:
    return os.environ[name]
  except KeyError:
    raise RuntimeError(f"{name} is not set; see README.md.") from None


def livekit_credentials() -> tuple[str, str, str]:
  """Returns the LiveKit URL, API key and API secret from the environment."""
  return (
      require_env("LIVEKIT_URL"),
      require_env("LIVEKIT_API_KEY"),
      require_env("LIVEKIT_API_SECRET"),
  )
