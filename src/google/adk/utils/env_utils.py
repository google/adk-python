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

"""Utilities for environment variable handling.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import os


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
