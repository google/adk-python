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
import pytest
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.events.event import Event


class DummyFlow(BaseLlmFlow):

  async def _run_one_step_async(self, invocation_context):
    # Simulate a truncated/partial response (e.g. max_token hit).
    yield Event(invocation_id="inv1", author="test", partial=True)


@pytest.mark.asyncio
async def test_run_async_breaks_on_partial_truncated():
  flow = DummyFlow()
  events = []
  # Collect events, but bail out if we ever get more than one.
  async for ev in flow.run_async(None):
    events.append(ev)
    if len(events) > 1:
      break

  # Before the fix, we'd collect at least 2 events (infinite loop).
  # After the fix, we should see exactly one truncated event, then stop.
  assert (
      len(events) == 1
  ), f"Expected run_async to stop after one partial event, got {len(events)}"
  assert events[0].partial is True
