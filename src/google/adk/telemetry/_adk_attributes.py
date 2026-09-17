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

ADK_EXPERIMENTAL_SKILL_NAME = 'adk.experimental.skill.name'
ADK_EXPERIMENTAL_SKILL_DESCRIPTION = 'adk.experimental.skill.description'
ADK_EXPERIMENTAL_SKILL_ADDITIONAL_TOOLS = (
    'adk.experimental.skill.additional_tools'
)
ADK_EXPERIMENTAL_SKILL_SOURCE_URI = 'adk.experimental.skill.source.uri'
ADK_EXPERIMENTAL_SKILL_RESOURCE_PATH = 'adk.experimental.skill.resource.path'
ADK_EXPERIMENTAL_SKILL_SCRIPT_PATH = 'adk.experimental.skill.script.path'
ADK_EXPERIMENTAL_SKILL_SCRIPT_ENDED_WITH_ERROR = (
    'adk.experimental.skill.script.ended_with_error'
)
ADK_EXPERIMENTAL_SKILL_SCRIPT_EXIT_CODE = (
    'adk.experimental.skill.script.exit_code'
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
