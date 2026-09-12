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

"""Utilities for environment variable handling.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import os
import warnings


def is_env_enabled(env_var_name: str, default: str = '0') -> bool:
  """Check if an environment variable is enabled.

  An environment variable is considered enabled if its value (case-insensitive)
  is 'true' or '1'.

  Args:
    env_var_name: The name of the environment variable to check.
    default: The default value to use if the environment variable is not set.
      Defaults to '0'.

  Returns:
    True if the environment variable is enabled, False otherwise.

  Examples:
    >>> os.environ['MY_FLAG'] = 'true'
    >>> is_env_enabled('MY_FLAG')
    True

    >>> os.environ['MY_FLAG'] = '1'
    >>> is_env_enabled('MY_FLAG')
    True

    >>> os.environ['MY_FLAG'] = 'false'
    >>> is_env_enabled('MY_FLAG')
    False

    >>> is_env_enabled('NONEXISTENT_FLAG')
    False

    >>> is_env_enabled('NONEXISTENT_FLAG', default='1')
    True
  """
  return os.environ.get(env_var_name, default).lower() in ['true', '1']


def is_enterprise_mode_enabled() -> bool:
  """Check if Google GenAI Enterprise mode is enabled via environment variables.

  On Google Cloud, unset project/location/enterprise flags are filled from the
  instance metadata server before this check (explicit env values still win).

  Returns:
    True if enabled, False otherwise.
  """
  # Ensure GCP metadata defaults are applied once before reading the flags so
  # Agent Engine / Cloud Run agents work without a `.env` for Vertex identity.
  apply_gcp_runtime_defaults()

  if 'GOOGLE_GENAI_USE_ENTERPRISE' in os.environ:
    return is_env_enabled('GOOGLE_GENAI_USE_ENTERPRISE')
  if 'GOOGLE_GENAI_USE_VERTEXAI' in os.environ:
    warnings.warn(
        'GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use'
        ' GOOGLE_GENAI_USE_ENTERPRISE instead',
        DeprecationWarning,
        stacklevel=2,
    )
    return is_env_enabled('GOOGLE_GENAI_USE_VERTEXAI')
  return False


def apply_gcp_runtime_defaults() -> dict[str, str]:
  """Apply GCP metadata defaults for unset Vertex/project/location env vars.

  When ADK runs on Google Cloud without an explicit `.env` / shell configuration,
  project id, location, and enterprise/Vertex mode are filled from the instance
  metadata server. Values already present in the environment are never
  overwritten.

  Returns:
    Mapping of environment variable names to values that were applied.
  """
  from ._gcp_metadata import apply_gcp_runtime_defaults as _apply

  return _apply()
