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

"""Lazy access to the optional `elevenlabs` SDK and to the API key."""

from __future__ import annotations

import os
from typing import Any
from typing import Optional

API_KEY_ENV = "ELEVENLABS_API_KEY"


def load_elevenlabs() -> Any:
  """Imports and returns the `elevenlabs` module."""
  try:
    import elevenlabs  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(
        "The ElevenLabs integration requires the ElevenLabs SDK. "
        'Install it with `pip install "google-adk[elevenlabs]"`.'
    ) from exc
  return elevenlabs


def resolve_api_key(api_key: Optional[str] = None) -> str:
  """Returns the explicit key, else `$ELEVENLABS_API_KEY`, else raises."""
  if api_key:
    return api_key

  key = os.environ.get(API_KEY_ENV)
  if not key:
    raise ValueError(
        f"{API_KEY_ENV} environment variable is not set. "
        f"Export it via `export {API_KEY_ENV}='your-api-key'`."
    )
  return key
