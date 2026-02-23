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

"""Regression tests for persisted event count assertions."""

from google.adk.agents.llm_agent import LlmAgent
import pytest

from .. import testing_utils


@pytest.mark.asyncio
async def test_run_async_event_counts_require_fresh_session_fetch() -> None:
  """Persisted event counts must be asserted against a freshly fetched session."""
  agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(responses=["first", "second"]),
  )
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  first_invocation_events = await runner.run_async("hello")
  stale_session = await runner.runner.session_service.get_session(
      app_name=runner.app_name,
      user_id="test_user",
      session_id=runner.session_id,
  )
  assert stale_session is not None

  second_invocation_events = await runner.run_async("hello again")

  # This session snapshot was retrieved before the second invocation and should
  # not be used for final persisted-event assertions.
  stale_count = len(stale_session.events)
  fresh_session = await runner.runner.session_service.get_session(
      app_name=runner.app_name,
      user_id="test_user",
      session_id=runner.session_id,
  )
  assert fresh_session is not None
  fresh_count = len(fresh_session.events)

  assert stale_count < fresh_count
  # Sanity check: the persisted delta covers at least all emitted events.
  assert fresh_count - stale_count >= len(second_invocation_events)
  assert len(first_invocation_events) > 0
