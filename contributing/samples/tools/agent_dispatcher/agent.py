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

"""Sample orchestrator that dispatches runtime agents with follow-ups.

Demonstrates AgentDispatcherToolset for issue #4759: spawn an arbitrary
child agent mid-run, then message it again on the same persistent session.
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.tools.agent_dispatcher import AgentDispatcherToolset


def lookup_fact(topic: str) -> dict[str, str]:
  """Return a tiny canned fact for the dispatched researcher."""
  facts = {
      'adk': "ADK is Google's Agent Development Kit for building agents.",
      'gemini': "Gemini is Google's family of multimodal models.",
  }
  key = topic.strip().lower()
  return {
      'topic': topic,
      'fact': facts.get(key, f'No canned fact for {topic}.'),
  }


dispatcher = AgentDispatcherToolset(
    model='gemini-2.5-flash',
    tool_allowlist={'lookup_fact': lookup_fact},
)

root_agent = Agent(
    name='orchestrator',
    model='gemini-2.5-flash',
    instruction="""\
You coordinate research by dispatching specialist agents.

When the user asks for research:
1. Call dispatch_agent with a clear name, instruction, and user_message.
   Optionally pass tool_names=["lookup_fact"] so the specialist can use it.
2. If you need more detail, call message_agent with the returned dispatch_id.
3. You may call get_agent_result to recall the latest result.

Keep the user updated with a concise final answer.
""",
    tools=[dispatcher],
)
