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

"""Utilities for telemetry.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from typing import Any

from .env_utils import is_env_enabled


def is_telemetry_enabled(obj: Any) -> bool:
  """Checks if telemetry is enabled.

  By default, telemetry is enabled unless any of an object's telemetry disabling
  variables are set to true.

  Args:
    obj: The object to check if telemetry is enabled for. It is expected
      to have a `disable_telemetry` boolean attribute.

  Returns:
      `False` if `OTEL_SDK_DISABLED` or `ADK_TELEMETRY_DISABLED` environment
      variables are set to a truthy value (e.g. 'true', '1'), or if
      `obj.disable_telemetry` is `True`. Otherwise, returns `True`.

  Examples:
      >>> import os
      >>> class MyObject:
      ...     disable_telemetry = False
      >>> my_obj = MyObject()
      >>> # Telemetry disabled by environment variable
      >>> os.environ['OTEL_SDK_DISABLED'] = 'true'
      >>> is_telemetry_enabled(my_obj)
      False
      >>> del os.environ['OTEL_SDK_DISABLED']
      >>> # Telemetry disabled by another environment variable
      >>> os.environ['ADK_TELEMETRY_DISABLED'] = '1'
      >>> is_telemetry_enabled(my_obj)
      False
      >>> del os.environ['ADK_TELEMETRY_DISABLED']
      >>> # Telemetry disabled by attribute
      >>> my_obj.disable_telemetry = True
      >>> is_telemetry_enabled(my_obj)
      False
      >>> # Telemetry enabled
      >>> my_obj.disable_telemetry = False
      >>> is_telemetry_enabled(my_obj)
      True
  """
  telemetry_disabled = (
      is_env_enabled("OTEL_SDK_DISABLED")
      or is_env_enabled("ADK_TELEMETRY_DISABLED")
      or getattr(obj, "disable_telemetry", False)
  )
  return not telemetry_disabled
