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

"""The serving tier a model call runs under."""

from __future__ import annotations

import enum


class ServiceTier(str, enum.Enum):
  """Serving tiers the interactions API offers for a model call.

  Mirrors `google.genai.interactions.ServiceTier` rather than importing it.
  That type is a `Literal` union with no members to reference, and the module
  it lives in is not otherwise loaded by an ordinary agent, so importing it
  here would add its cost to every ADK start for the sake of an annotation.

  Subclasses `str`, so a member compares and serializes exactly like the
  string it carries and can be sent to the API as-is. Fields typed with this
  enum also accept a plain string, so a tier the backend adds before ADK
  learns about it still works.

  Kept in its own module, like `StreamingMode`, because the config, the
  request and the interactions transport are all typed with it.
  """

  FLEX = 'flex'
  """Best-effort capacity at a lower cost, with no latency guarantee."""

  STANDARD = 'standard'
  """The default tier."""

  PRIORITY = 'priority'
  """Reserved capacity for latency-sensitive calls."""

  DEFERRED = 'deferred'
  """Queued to run on off-peak capacity.

  The call waits for room instead of being turned away when capacity is
  tight. The API returns an interaction id as soon as the work is accepted
  rather than a result, so this tier cannot be combined with
  `StreamingMode.SSE`.
  """
