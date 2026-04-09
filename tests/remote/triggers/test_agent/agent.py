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

"""Echo agent for remote trigger integration tests.

Uses a before_model_callback to echo user input without calling the LLM,
making tests fast, deterministic, and free of LLM model quota usage.

Supports optional 429 simulation via the SIMULATE_429_COUNT environment
variable: when set to N > 0, the first N invocations per session will raise
a RuntimeError containing "429 RESOURCE_EXHAUSTED" before succeeding.
"""

import os

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# Track 429 simulation state across invocations within a process.
# Keyed by session_id to allow per-session failure counts.
_429_counter: dict[str, int] = {}


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse:
  """Echo user input back without calling the LLM.

  If SIMULATE_429_COUNT is set, raises a transient error for the first N
  invocations (per session) to exercise the retry-with-backoff logic in
  the trigger endpoints.
  """
  fail_count = int(os.environ.get("SIMULATE_429_COUNT", "0"))
  session = callback_context.session
  session_id = session.id if session else "default"

  if fail_count > 0:
    current = _429_counter.get(session_id, 0)
    if current < fail_count:
      _429_counter[session_id] = current + 1
      raise RuntimeError("429 RESOURCE_EXHAUSTED: simulated quota exceeded")

  # Extract the most recent user message from the session events.
  user_text = ""
  if session and session.events:
    for event in reversed(session.events):
      if event.content and event.content.role == "user" and event.content.parts:
        user_text = event.content.parts[0].text or ""
        break

  # Fall back to the current LLM request contents if no session event found.
  if not user_text and llm_request.contents:
    for content in reversed(llm_request.contents):
      if content.role == "user" and content.parts:
        user_text = content.parts[0].text or ""
        break

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(text=f"ECHO: {user_text}")],
      ),
  )


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="trigger_echo_agent",
    instruction="Echo agent for trigger testing.",
    before_model_callback=before_model_callback,
)
