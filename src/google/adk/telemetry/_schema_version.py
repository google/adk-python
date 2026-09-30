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

"""Opt-in for the ADK telemetry schema version.

``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN`` lets a deployment pin which version of
the ADK telemetry format (span names, span/log attributes, metrics) it emits.

Why this exists:

* **Staged migration.** Telemetry format changes are breaking for anyone who
  built dashboards/alerts on the previous shape. The version env lets users
  fall back to the legacy format if they rely on it, and migrate on their own
  schedule.
* **Fast iteration on managed services.** GCP-managed runtimes (e.g. Agent
  Runtime / Vertex Agent Engine) can pin themselves to the latest version and
  iterate quickly, decoupled from the broader user base.
* **Eventually removable.** Ideally this knob is phased out once ADK is
  ({almost,} fully) OTel semconv compliant, after which we no longer expect
  breaking telemetry changes.

Default resolution:

* ``2`` on Agent Engine, detected via the presence of the
  ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` env var.
* ``1`` everywhere else.

Migration plan (steps land incrementally; this comment should be kept in sync
as each one ships):

1. The ``invocation`` span is updated to the ``invoke_workflow`` span.
2. The ``call_llm`` span is removed.
3. The ``execute_tool`` content-bearing attributes become OTel semconv aligned.
4. The experimental OTel GenAI semconv becomes the default in both
   ``opentelemetry-instrumentation-google-genai`` and ADK's native
   instrumentation.
5. ``2`` becomes the global default (this knob flips).
6. After ~6 months, the env var is phased out entirely, along with support for
   the LEGACY schema.
"""

from __future__ import annotations

import os

# Env var users set to pin the ADK telemetry schema version ("1" or "2").
ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"

# Presence of this env var indicates the process runs on Vertex Agent Engine.
GOOGLE_CLOUD_AGENT_ENGINE_ID = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

# Legacy telemetry format: top-level ``invocation`` span, no entrypoint
# ``invoke_workflow`` span/metric.
SCHEMA_VERSION_LEGACY = 1

# OTel-semconv-aligned telemetry format: the ``invocation`` span is replaced by
# an entrypoint ``invoke_workflow {entrypoint}`` span + duration metric.
SCHEMA_VERSION_SEMCONV_ALIGNED = 2


def resolve_schema_version() -> int:
  """Resolves the active ADK telemetry schema version.

  Precedence: ``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN`` (if set to a recognized
  value) > ``2`` on Agent Engine > ``1``.

  Returns:
    Either :data:`SCHEMA_VERSION_LEGACY` or
    :data:`SCHEMA_VERSION_SEMCONV_ALIGNED`.
  """
  opt_in = os.environ.get(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, "").strip()
  if opt_in == "1":
    return SCHEMA_VERSION_LEGACY
  if opt_in == "2":
    return SCHEMA_VERSION_SEMCONV_ALIGNED

  # Unset/unrecognized: default to v2 on Agent Engine, v1 elsewhere.
  if os.environ.get(GOOGLE_CLOUD_AGENT_ENGINE_ID):
    return SCHEMA_VERSION_SEMCONV_ALIGNED
  return SCHEMA_VERSION_LEGACY
