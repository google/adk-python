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

"""Request processor that runs token-threshold event compaction."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ...apps.compaction import _has_token_threshold_config
from ...apps.compaction import _run_compaction_for_token_threshold_config
from ...events.event import Event
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest


class CompactionRequestProcessor(BaseLlmRequestProcessor):
  """Compacts session events before contents are prepared for model calls."""

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del llm_request
    config = invocation_context.events_compaction_config
    if not _has_token_threshold_config(config):
      return
      yield  # Required for AsyncGenerator.

    token_compacted = await _run_compaction_for_token_threshold_config(
        config=config,
        session=invocation_context.session,
        session_service=invocation_context.session_service,
        agent=invocation_context.agent,
        agent_name=invocation_context.agent.name,
        current_branch=invocation_context.branch,
    )
    if token_compacted:
      invocation_context.token_compaction_checked = True
    return
    yield  # Required for AsyncGenerator.


request_processor = CompactionRequestProcessor()
