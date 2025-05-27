from typing import Any
from typing import Optional

from google.adk.sessions import InMemorySessionService
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry import trace_call_llm
from google.genai import types
import pytest


async def _create_invocation_context(
    agent: LlmAgent, state: Optional[dict[str, Any]] = None
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return invocation_context


@pytest.mark.asyncio
async def test_trace_call_llm_function_response_includes_part_from_bytes():
  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
    contents=[
      types.Content(
        role="user",
        parts=[
          types.Part.from_function_response(
            name="test_function_1",
            response={
              "result": b"test_data",
            },
          ),
        ],
      ),
      types.Content(
        role="user",
        parts=[
          types.Part.from_function_response(
            name="test_function_2",
            response={
              "result": types.Part.from_bytes(data=b"test_data", mime_type="application/octet-stream"),
            },
          ),
        ],
      ),
    ],
    config=types.GenerateContentConfig(system_instruction=""),
  )
  llm_response = LlmResponse(turn_complete=True)
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)
