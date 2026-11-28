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

"""ADK-owned span attribute names.

These attributes are defined by ADK itself; they are not part of any
OpenTelemetry semantic convention (neither the stable one in
``_stable_semconv`` nor the experimental one in ``_experimental_semconv``).

Everything named ``adk.experimental.*`` is emitted only when experimental
telemetry is enabled and carries no compatibility guarantee: an attribute may be
renamed, restructured, or removed in any release.
"""

from __future__ import annotations

ADK_SKILL_ADDITIONAL_TOOLS = 'adk.skill.additional_tools'
ADK_EXPERIMENTAL_SKILL_SCRIPT_ENDED_WITH_ERROR = (
    'adk.experimental.skill.script.ended_with_error'
)

ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT = 'adk.experimental.context_cache.hit'
ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT = (
    'adk.experimental.context_cache.fingerprint'
)
ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT = (
    'adk.experimental.context_cache.contents_count'
)
ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED = (
    'adk.experimental.context_cache.invocations_used'
)

# Names the callback that produced the response the span records.
ADK_EXPERIMENTAL_RESPONSE_SOURCE = 'adk.experimental.response.source'

# Tool confirmation decision, recorded as a span event on the execute_tool
# span (see telemetry.tracing.trace_tool_confirmation_decision). These names
# are ADK-owned and provisional: how an authorization decision is represented
# is still being discussed in the OpenTelemetry GenAI semantic conventions,
# and these names will be aligned with that convention when it settles.
TOOL_CONFIRMATION_DECISION_EVENT = 'adk.tool_confirmation.decision'
ADK_TOOL_CONFIRMATION_CONFIRMED = 'adk.tool_confirmation.confirmed'
ADK_TOOL_CONFIRMATION_ARGUMENTS_HASH = 'adk.tool_confirmation.arguments_hash'
ADK_TOOL_CONFIRMATION_CONFIRMED_BY = 'adk.tool_confirmation.confirmed_by'

ADK_EXPERIMENTAL_GROUNDING_QUERY_COUNT = (
    'adk.experimental.grounding.query_count'
)
ADK_EXPERIMENTAL_GROUNDING_CHUNK_COUNT = (
    'adk.experimental.grounding.chunk_count'
)
ADK_EXPERIMENTAL_GROUNDING_WEB_SEARCH_QUERIES = (
    'adk.experimental.grounding.web_search_queries'
)
ADK_EXPERIMENTAL_GROUNDING_SOURCE_URIS = (
    'adk.experimental.grounding.source_uris'
)
