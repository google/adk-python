import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.runners import StopSignal
from google.genai.types import Content
from google.genai.types import Part
import pytest


# Used to tell the main loop when to interrupt the run_async
class TriggerSignal:

  def __init__(self):
    self.triggered = False

  def set(self):
    self.triggered = True

  def reset(self):
    self.triggered = False

  def is_set(self) -> bool:
    return self.triggered


# Mocks an LLM, outputting events every 1s
class MockAgent(BaseAgent):
  name: str = "MockAgent"

  async def run_async(self, ctx: Any) -> AsyncGenerator[Event, None]:
    cycle = 0

    while True:
      await asyncio.sleep(1)
      cycle += 1
      yield Event(
          author="model",
          content=Content(role="model", parts=[Part(text=f"{cycle}")]),
      )


APP_NAME = "adk_test_app"
USER_ID = "adk_test_user"
STOP_SIGNAL = StopSignal()
TRIGGER_SIGNAL = TriggerSignal()


@pytest.mark.asyncio
async def test_stop_signal():
  async def consume_stream(runner, trigger_signal, **kwargs):
    cycle = 0

    async for event in runner.run_async(**kwargs):
      cycle += 1
      content = event.content.parts[0].text

      if not content:
        # The fourth cycle should be interrupted
        # Only interrupted cycles would not have content objects given this MockAgent
        assert cycle == 4
        assert event.interrupted
      if content and content == "3":
        # Tell main loop to interrupt the fourth cycle
        trigger_signal.set()

  client = MockAgent()
  runner = InMemoryRunner(agent=client, app_name=APP_NAME)

  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  message = Content(role="user", parts=[Part(text="Necessary message")])

  task = asyncio.create_task(
      consume_stream(
          runner=runner,
          trigger_signal=TRIGGER_SIGNAL,
          session_id=session.id,
          user_id=USER_ID,
          new_message=message,
          stop_signal=STOP_SIGNAL,
          run_config=RunConfig(streaming_mode=StreamingMode.SSE),
      )
  )

  # Wait for 3 events to pass
  while not TRIGGER_SIGNAL.is_set():
    await asyncio.sleep(0.1)

  # Interrupt the run_async
  STOP_SIGNAL.stop()
