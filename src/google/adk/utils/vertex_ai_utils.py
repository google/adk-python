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

"""Utilities for Vertex AI. Includes helper functions for Express Mode.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import os
from typing import Optional

from ..utils.env_utils import is_env_enabled


def get_express_mode_api_key(
    project: Optional[str],
    location: Optional[str],
    express_mode_api_key: Optional[str],
) -> Optional[str]:
  """Validates and returns the API key for Express Mode."""
  if (project or location) and express_mode_api_key:
    raise ValueError(
        'Cannot specify project or location and express_mode_api_key. '
        'Either use project and location, or just the express_mode_api_key.'
    )
  if is_env_enabled('GOOGLE_GENAI_USE_VERTEXAI'):
    return express_mode_api_key or os.environ.get('GOOGLE_API_KEY', None)
  else:
    return None
