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

from typing import TYPE_CHECKING

from .env_utils import is_env_enabled

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent


def is_telemetry_enabled(agent: "BaseAgent") -> bool:
  """Check if telemetry is enabled for the given agent.

    By default telemetry is enabled for an agent unless any of the variables to disable telemetry are set to true.

  Args:
    agent: The agent to check if telemetry is enabled for.

  Returns:
      False if any of the environment variables or attributes to disable telemetry are set to True, 'true' or 1, False otherwise.

  Examples:
      >>> os.environ['OTEL_SDK_DISABLED'] = 'true'
      >>> is_telemetry_enabled(my_agent)
      False

      >>> os.environ['ADK_TELEMETRY_DISABLED'] = 1
      >>> is_telemetry_enabled(my_agent)
      False

      >>> my_agent.disable_telemetry = True
      >>> is_telemetry_enabled(my_agent)
      False

      >>> os.environ['OTEL_SDK_DISABLED'] = 1
      >>> os.environ['ADK_TELEMETRY_DISABLED'] = 'false'
      >>> my_agent.disable_telemetry = False
      >>> is_telemetry_enabled(my_agent)
      False

      >>> os.environ['OTEL_SDK_DISABLED'] = 'false'
      >>> os.environ['ADK_TELEMETRY_DISABLED'] = 0
      >>> my_agent.disable_telemetry = False
      >>> is_telemetry_enabled(my_agent)
      True
  """
  telemetry_disabled = (
      is_env_enabled("OTEL_SDK_DISABLED")
      or is_env_enabled("ADK_TELEMETRY_DISABLED")
      or getattr(agent, "disable_telemetry", False)
  )
  return not telemetry_disabled
