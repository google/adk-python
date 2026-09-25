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

"""Context variables for internal telemetry use."""

from __future__ import annotations

import contextvars
from typing import Optional

# Internal context variable for Visual Builder usage tracking.
# True if the current execution is within a Visual Builder context.
_is_visual_builder: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_is_visual_builder", default=False
)

# Internal context variable for caller-surface telemetry attribution.
# Read at BigQueryAgentAnalyticsPlugin construction time, or (on the default
# per-instance shared loop, if unset at construction) when the loop state is
# first built and pinned for that (plugin, loop) pair thereafter. When set
# (e.g. "my-surface"), the plugin stamps
# ``google-adk-bq-logger-{surface}/{version}`` into the BigQuery Write API
# ``trace_id`` (taking precedence over ``_is_visual_builder``) and
# ``google-adk-{surface}/{version}`` into the gRPC user agent.
# Callers are responsible for passing a clean token matching
# ``[a-z0-9]([a-z0-9-]*[a-z0-9])?`` (no spaces/colons; avoid ``"bq-logger"``,
# ``"java"``, and ``"visual-builder"``, which already occupy the namespace).
# The value must also stay constant for the lifetime of any plugin instance
# constructed while it is set: the plugin captures it at ``__init__``, and
# on a shared loop, if ``__init__`` captured nothing, ``_build_loop_state``
# pins whatever is visible when the loop state is first built. Flipping the
# value mid-flight does not re-attribute an already-built writer.
_telemetry_surface: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("_telemetry_surface", default=None)
)


def _get_telemetry_surface() -> Optional[str]:
  """Returns the active caller-surface label.

  Unifies the explicit ``_telemetry_surface`` token with the legacy
  ``_is_visual_builder`` flag, which maps to ``"visual-builder"``.
  """
  return _telemetry_surface.get() or (
      "visual-builder" if _is_visual_builder.get() else None
  )
