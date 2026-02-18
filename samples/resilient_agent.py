# Sample: Using LlmResiliencePlugin for robust model calls
#
# Run with:
#   PYTHONPATH=$(pwd)/src python samples/resilient_agent.py
#
# This demonstrates:
# - Configuring LlmResiliencePlugin for retries and fallbacks
# - Running a minimal in-memory agent with a mocked model

from __future__ import annotations

import asyncio

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.plugins.llm_resilience_plugin import LlmResiliencePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


class DemoFailThenSucceedModel(BaseLlm):
  model: str = "demo-fail-succeed"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._attempts: int = 0  # Instance variable for proper state management

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["demo-fail-succeed"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ):
    # Fail for the first attempt, then succeed
    self._attempts += 1
    if self._attempts < 2:
      raise TimeoutError("Simulated transient failure")
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Recovered on retry!")],
        ),
        partial=False,
    )


# Register test models
LLMRegistry.register(DemoFailThenSucceedModel)


async def main():
  # Agent with the failing-then-succeed model
  agent = LlmAgent(name="resilient_agent", model="demo-fail-succeed")

  # Build services and runner in-memory
  artifact_service = InMemoryArtifactService()
  session_service = InMemorySessionService()
  memory_service = InMemoryMemoryService()

  runner = Runner(
      app_name="resilience_demo",
      agent=agent,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      plugins=[
          LlmResiliencePlugin(
              max_retries=2,
              backoff_initial=0.1,
              backoff_multiplier=2.0,
              jitter=0.1,
              fallback_models=["mock"],  # Demonstration; not used here
          )
      ],
  )

  # Create a session and run once
  session = await session_service.create_session(
      app_name="resilience_demo", user_id="demo"
  )
  events = []
  async for ev in runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
  ):
    events.append(ev)

  print("Collected", len(events), "events")
  for e in events:
    if e.content and e.content.parts and e.content.parts[0].text:
      print("MODEL:", e.content.parts[0].text.strip())


if __name__ == "__main__":
  asyncio.run(main())
