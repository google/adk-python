"""Integration test for output_schema + tools behavior.

Requires GOOGLE_API_KEY or Vertex AI credentials.
Run with: python -m pytest tests/integration/test_output_schema_with_tools.py -v -s
"""

import os
import time

from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
import pytest


class AnalysisResult(BaseModel):
  summary: str = Field(description='Brief summary of the analysis')
  confidence: float = Field(description='Confidence score between 0 and 1')


def search_data(query: str) -> str:
  """Search for data based on the query."""
  return f'Found data for: {query}. Revenue is $1M, growth is 15%.'


def calculate_metric(metric_name: str, value: float) -> str:
  """Calculate a business metric."""
  return f'{metric_name}: {value * 1.1:.2f} (adjusted)'


# Skip if no API key is configured.
skip_no_api_key = pytest.mark.skipif(
    not os.environ.get('GOOGLE_API_KEY')
    and not os.environ.get('GOOGLE_GENAI_USE_VERTEXAI'),
    reason='No Gemini API key or Vertex AI configured',
)


@skip_no_api_key
@pytest.mark.asyncio
async def test_basemodel_schema_with_tools():
  """Test that BaseModel output_schema + tools produces structured output."""
  agent = LlmAgent(
      name='analyst',
      model='gemini-2.5-flash',
      instruction=(
          'Analyze the query using the available tools, then return'
          ' structured output.'
      ),
      output_schema=AnalysisResult,
      tools=[search_data, calculate_metric],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      agent=agent, app_name='test_app', session_service=session_service
  )
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  events = []
  start = time.time()

  async for event in runner.run_async(
      user_id='test_user',
      session_id=session.id,
      new_message=types.Content(
          role='user',
          parts=[types.Part(text='Analyze Q1 revenue performance')],
      ),
  ):
    events.append(event)

  elapsed = time.time() - start

  # Should complete within a reasonable time (not infinite loop).
  assert elapsed < 120, f'Took {elapsed:.1f}s — possible infinite loop'

  # Should have at least one event with structured output.
  final_texts = [
      e.content.parts[0].text
      for e in events
      if e.content and e.content.parts and e.content.parts[0].text
  ]
  assert len(final_texts) > 0, 'No text output produced'
  print(f'\nCompleted in {elapsed:.1f}s with {len(events)} events')
  print(f'Final output: {final_texts[-1][:200]}')


@skip_no_api_key
@pytest.mark.asyncio
async def test_str_schema_with_tools():
  """Test that str output_schema + tools produces output (not infinite loop)."""
  agent = LlmAgent(
      name='analyst',
      model='gemini-2.5-flash',
      instruction='Search for the data, then provide a brief text summary.',
      output_schema=str,
      tools=[search_data],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      agent=agent, app_name='test_app', session_service=session_service
  )
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  events = []
  start = time.time()

  async for event in runner.run_async(
      user_id='test_user',
      session_id=session.id,
      new_message=types.Content(
          role='user',
          parts=[types.Part(text='What is the Q1 revenue?')],
      ),
  ):
    events.append(event)

  elapsed = time.time() - start

  assert elapsed < 120, f'Took {elapsed:.1f}s — possible infinite loop'
  assert len(events) > 0, 'No events produced'
  print(f'\nCompleted in {elapsed:.1f}s with {len(events)} events')
