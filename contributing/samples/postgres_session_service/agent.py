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

"""Sample agent demonstrating PostgreSQL session persistence."""

from datetime import datetime
from datetime import timezone

from google.adk.agents.llm_agent import Agent


def get_current_time() -> str:
  """Get the current time.

  Returns:
    A string with the current time in ISO 8601 format.
  """
  return datetime.now(timezone.utc).isoformat()


root_agent = Agent(
    model="gemini-2.0-flash",
    name="postgres_session_agent",
    description="A sample agent demonstrating PostgreSQL session persistence.",
    instruction="""
      You are a helpful assistant that demonstrates session persistence.
      You can remember previous conversations within the same session.
      Use the get_current_time tool when asked about the time.
      When the user asks what you remember, summarize the previous conversation.
    """,
    tools=[get_current_time],
)
