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

``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN`` pins which version of the ADK telemetry
format (span names, span/log attributes, metrics) a deployment emits:

* ``2`` or ``otel_semconv_1_44`` (default): OpenTelemetry GenAI semconv
  aligned. ``invoke_workflow`` replaces ``invocation``, there is no
  ``call_llm`` span, tool content is ``gen_ai.tool.call.*``, and the
  experimental GenAI semconv is the default.
* ``1`` or ``otel_semconv_1_36``: the legacy format, with the stable GenAI
  semconv recorded by ADK even when
  ``opentelemetry-instrumentation-google-genai`` is installed.

``1`` is supported until at least 2027-03-31, then removed with this knob.
"""

from __future__ import annotations

import os

ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"

SCHEMA_VERSION_LEGACY = 1
SCHEMA_VERSION_SEMCONV_ALIGNED = 2


def resolve_schema_version() -> int:
  """Returns 1 if ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN pins it, else 2."""
  opt_in = os.environ.get(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, "").strip()
  # `otel_semconv_1_44` needs no branch, it resolves like the default.
  if opt_in in ("1", "otel_semconv_1_36"):
    return SCHEMA_VERSION_LEGACY
  return SCHEMA_VERSION_SEMCONV_ALIGNED
