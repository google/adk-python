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

"""Private runner and execution utility functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..plugins.plugin_manager import PluginManager

logger = logging.getLogger("google_adk." + __name__)


async def _notify_run_error(
    plugin_manager: PluginManager,
    invocation_context: InvocationContext,
    error: Exception,
) -> None:
  """Best-effort on_run_error notification; never masks the original error.

  on_run_error_callback is notification-only: the triggering exception is
  always re-raised by the caller, so any exception from the callback itself
  (or from a test double that does not implement it) is logged and suppressed.
  """
  try:
    await plugin_manager.run_on_run_error_callback(
        invocation_context=invocation_context, error=error
    )
  except Exception:  # pylint: disable=broad-except
    logger.exception(
        "on_run_error_callback raised; suppressing so the original run error"
        " propagates."
    )
