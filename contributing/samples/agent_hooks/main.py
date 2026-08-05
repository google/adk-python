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

"""Runs the support agent with agent-hooks governance enabled.

Prerequisites (see README.md):
  * Ollama running locally with the ``qwen2.5:7b`` model pulled.
  * ``pip install "google-adk[agent-hooks]" litellm``

Run:
  python -m contributing.samples.agent_hooks.main
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.apps.app import App
from google.adk.plugins import AgentHooksPlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent import root_agent
from .governance import ToolGovernanceInterceptor

_APP_NAME = "agent_hooks_demo"


async def main() -> None:
  """Runs two prompts: one benign (redacted), one destructive (denied)."""
  # ``record_sink`` receives an auditable InterceptionRecord per decision.
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[ToolGovernanceInterceptor()],
      record_sink=records.append,
  )

  app = App(name=_APP_NAME, root_agent=root_agent, plugins=[plugin])
  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      user_id="user", app_name=_APP_NAME
  )

  prompts = [
      "Look up the account details for user 42.",
      "Now delete account 42.",
  ]
  for prompt in prompts:
    print(f"\n=== USER: {prompt} ===")
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        ),
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            print(f"[{event.author}] {part.text}")
          if part.function_call:
            print(f"[{event.author}] -> tool call: {part.function_call.name}")
          if part.function_response:
            print(
                f"[{event.author}] <- tool result:"
                f" {part.function_response.response}"
            )

  print("\n=== agent-hooks audit trail ===")
  for record in records:
    verdict = record.verdict
    print(
        f"seq={record.sequence:<2} {record.interception_point.value:<16}"
        f" -> {verdict.decision.value}"
        + (f" ({verdict.reason})" if verdict.reason else "")
    )


if __name__ == "__main__":
  asyncio.run(main())
