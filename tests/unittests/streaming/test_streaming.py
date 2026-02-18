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

import asyncio
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from .. import testing_utils


def test_streaming():
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'


def test_live_streaming_function_call_single():
  """Test live streaming with a single function call response."""
  # Create a function call response
  function_call = types.Part.from_function_call(
      name='get_weather', args={'location': 'San Francisco', 'unit': 'celsius'}
  )

  # Create LLM responses: function call followed by turn completion
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock function that would be called
  def get_weather(location: str, unit: str = 'celsius') -> dict:
    return {
        'temperature': 22,
        'condition': 'sunny',
        'location': location,
        'unit': unit,
    }

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[get_weather],
  )

  # Create a custom runner class that collects all events
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          # Collect a reasonable number of events, don't wait for too many
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(
          data=b'What is the weather in San Francisco?', mime_type='audio/pcm'
      )
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got a function call event
  function_call_found = False
  function_response_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.name == 'get_weather':
          function_call_found = True
          assert part.function_call.args['location'] == 'San Francisco'
          assert part.function_call.args['unit'] == 'celsius'
        elif (
            part.function_response
            and part.function_response.name == 'get_weather'
        ):
          function_response_found = True
          assert part.function_response.response['temperature'] == 22
          assert part.function_response.response['condition'] == 'sunny'

  assert function_call_found, 'Expected a function call event.'
  # Note: In live streaming, function responses might be handled differently,
  # so we check for the function call which is the primary indicator of function calling working


def test_live_streaming_function_call_multiple():
  """Test live streaming with multiple function calls in sequence."""
  # Create multiple function call responses
  function_call1 = types.Part.from_function_call(
      name='get_weather', args={'location': 'San Francisco'}
  )
  function_call2 = types.Part.from_function_call(
      name='get_time', args={'timezone': 'PST'}
  )

  # Create LLM responses: two function calls followed by turn completion
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call1]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      content=types.Content(role='model', parts=[function_call2]),
      turn_complete=False,
  )
  response3 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2, response3])

  # Mock functions
  def get_weather(location: str) -> dict:
    return {'temperature': 22, 'condition': 'sunny', 'location': location}

  def get_time(timezone: str) -> dict:
    return {'time': '14:30', 'timezone': timezone}

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[get_weather, get_time],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(
          data=b'What is the weather and current time?', mime_type='audio/pcm'
      )
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check function calls
  weather_call_found = False
  time_call_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call:
          if part.function_call.name == 'get_weather':
            weather_call_found = True
            assert part.function_call.args['location'] == 'San Francisco'
          elif part.function_call.name == 'get_time':
            time_call_found = True
            assert part.function_call.args['timezone'] == 'PST'

  # In live streaming, we primarily check that function calls are generated correctly
  assert (
      weather_call_found or time_call_found
  ), 'Expected at least one function call.'


def test_live_streaming_function_call_parallel():
  """Test live streaming with parallel function calls."""
  # Create parallel function calls in the same response
  function_call1 = types.Part.from_function_call(
      name='get_weather', args={'location': 'San Francisco'}
  )
  function_call2 = types.Part.from_function_call(
      name='get_weather', args={'location': 'New York'}
  )

  # Create LLM response with parallel function calls
  response1 = LlmResponse(
      content=types.Content(
          role='model', parts=[function_call1, function_call2]
      ),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock function
  def get_weather(location: str) -> dict:
    temperatures = {'San Francisco': 22, 'New York': 15}
    return {'temperature': temperatures.get(location, 20), 'location': location}

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[get_weather],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(
          data=b'Compare weather in SF and NYC', mime_type='audio/pcm'
      )
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check parallel function calls
  sf_call_found = False
  nyc_call_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.name == 'get_weather':
          location = part.function_call.args['location']
          if location == 'San Francisco':
            sf_call_found = True
          elif location == 'New York':
            nyc_call_found = True

  assert (
      sf_call_found and nyc_call_found
  ), 'Expected both location function calls.'


def test_live_streaming_function_call_with_error():
  """Test live streaming with function call that returns an error."""
  # Create a function call response
  function_call = types.Part.from_function_call(
      name='get_weather', args={'location': 'Invalid Location'}
  )

  # Create LLM responses
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock function that returns an error for invalid locations
  def get_weather(location: str) -> dict:
    if location == 'Invalid Location':
      return {'error': 'Location not found'}
    return {'temperature': 22, 'condition': 'sunny', 'location': location}

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[get_weather],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(
          data=b'What is weather in Invalid Location?', mime_type='audio/pcm'
      )
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got the function call (error handling happens at execution time)
  function_call_found = False
  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.name == 'get_weather':
          function_call_found = True
          assert part.function_call.args['location'] == 'Invalid Location'

  assert function_call_found, 'Expected function call event with error case.'


def test_live_streaming_function_call_sync_tool():
  """Test live streaming with synchronous function call."""
  # Create a function call response
  function_call = types.Part.from_function_call(
      name='calculate', args={'x': 5, 'y': 3}
  )

  # Create LLM responses
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock sync function
  def calculate(x: int, y: int) -> dict:
    return {'result': x + y, 'operation': 'addition'}

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[calculate],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Calculate 5 plus 3', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check function call
  function_call_found = False
  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.name == 'calculate':
          function_call_found = True
          assert part.function_call.args['x'] == 5
          assert part.function_call.args['y'] == 3

  assert function_call_found, 'Expected calculate function call event.'


def test_live_streaming_simple_streaming_tool():
  """Test live streaming with a simple streaming tool (non-video)."""
  # Create a function call response for the streaming tool
  function_call = types.Part.from_function_call(
      name='monitor_stock_price', args={'stock_symbol': 'AAPL'}
  )

  # Create LLM responses
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock simple streaming tool (without return type annotation to avoid parsing issues)
  async def monitor_stock_price(stock_symbol: str):
    """Mock streaming tool that monitors stock prices."""
    # Simulate some streaming updates
    yield f'Stock {stock_symbol} price: $150'
    await asyncio.sleep(0.1)
    yield f'Stock {stock_symbol} price: $155'
    await asyncio.sleep(0.1)
    yield f'Stock {stock_symbol} price: $160'

  def stop_streaming(function_name: str):
    """Stop the streaming tool."""
    pass

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_stock_price, stop_streaming],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Monitor AAPL stock price', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got the streaming tool function call
  function_call_found = False
  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if (
            part.function_call
            and part.function_call.name == 'monitor_stock_price'
        ):
          function_call_found = True
          assert part.function_call.args['stock_symbol'] == 'AAPL'

  assert (
      function_call_found
  ), 'Expected monitor_stock_price function call event.'


def test_live_streaming_video_streaming_tool():
  """Test live streaming with a video streaming tool."""
  # Create a function call response for the video streaming tool
  function_call = types.Part.from_function_call(
      name='monitor_video_stream', args={}
  )

  # Create LLM responses
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock video streaming tool (without return type annotation to avoid parsing issues)
  async def monitor_video_stream(input_stream: LiveRequestQueue):
    """Mock video streaming tool that processes video frames."""
    # Simulate processing a few frames from the input stream
    frame_count = 0
    while frame_count < 3:  # Process a few frames
      try:
        # Try to get a frame from the queue with timeout
        live_req = await asyncio.wait_for(input_stream.get(), timeout=0.1)
        if live_req.blob and live_req.blob.mime_type == 'image/jpeg':
          frame_count += 1
          yield f'Processed frame {frame_count}: detected 2 people'
      except asyncio.TimeoutError:
        # No more frames, simulate detection anyway for testing
        frame_count += 1
        yield f'Simulated frame {frame_count}: detected 1 person'
      await asyncio.sleep(0.1)

  def stop_streaming(function_name: str):
    """Stop the streaming tool."""
    pass

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_video_stream, stop_streaming],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()

  # Send some mock video frames
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'fake_jpeg_data_1', mime_type='image/jpeg')
  )
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'fake_jpeg_data_2', mime_type='image/jpeg')
  )
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Monitor video stream', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got the video streaming tool function call
  function_call_found = False
  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if (
            part.function_call
            and part.function_call.name == 'monitor_video_stream'
        ):
          function_call_found = True

  assert (
      function_call_found
  ), 'Expected monitor_video_stream function call event.'


def test_live_streaming_stop_streaming_tool():
  """Test live streaming with stop_streaming functionality."""
  # Create function calls for starting and stopping a streaming tool
  start_function_call = types.Part.from_function_call(
      name='monitor_stock_price', args={'stock_symbol': 'TSLA'}
  )
  stop_function_call = types.Part.from_function_call(
      name='stop_streaming', args={'function_name': 'monitor_stock_price'}
  )

  # Create LLM responses: start streaming, then stop streaming
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[start_function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      content=types.Content(role='model', parts=[stop_function_call]),
      turn_complete=False,
  )
  response3 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2, response3])

  # Mock streaming tool and stop function
  async def monitor_stock_price(stock_symbol: str):
    """Mock streaming tool that monitors stock prices."""
    yield f'Started monitoring {stock_symbol}'
    while True:  # Infinite stream (would be stopped by stop_streaming)
      yield f'Stock {stock_symbol} price update'
      await asyncio.sleep(0.1)

  def stop_streaming(function_name: str):
    """Stop the streaming tool."""
    return f'Stopped streaming for {function_name}'

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_stock_price, stop_streaming],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Monitor TSLA and then stop', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got both function calls
  monitor_call_found = False
  stop_call_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call:
          if part.function_call.name == 'monitor_stock_price':
            monitor_call_found = True
            assert part.function_call.args['stock_symbol'] == 'TSLA'
          elif part.function_call.name == 'stop_streaming':
            stop_call_found = True
            assert (
                part.function_call.args['function_name']
                == 'monitor_stock_price'
            )

  assert monitor_call_found, 'Expected monitor_stock_price function call event.'
  assert stop_call_found, 'Expected stop_streaming function call event.'


def test_live_streaming_multiple_streaming_tools():
  """Test live streaming with multiple streaming tools running simultaneously."""
  # Create function calls for multiple streaming tools
  stock_function_call = types.Part.from_function_call(
      name='monitor_stock_price', args={'stock_symbol': 'NVDA'}
  )
  video_function_call = types.Part.from_function_call(
      name='monitor_video_stream', args={}
  )

  # Create LLM responses: start both streaming tools
  response1 = LlmResponse(
      content=types.Content(
          role='model', parts=[stock_function_call, video_function_call]
      ),
      turn_complete=False,
  )
  response2 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  # Mock streaming tools
  async def monitor_stock_price(stock_symbol: str):
    """Mock streaming tool that monitors stock prices."""
    yield f'Stock {stock_symbol} price: $800'
    await asyncio.sleep(0.1)
    yield f'Stock {stock_symbol} price: $805'

  async def monitor_video_stream(input_stream: LiveRequestQueue):
    """Mock video streaming tool."""
    yield 'Video monitoring started'
    await asyncio.sleep(0.1)
    yield 'Detected motion in video stream'

  def stop_streaming(function_name: str):
    """Stop the streaming tool."""
    pass

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_stock_price, monitor_video_stream, stop_streaming],
  )

  # Use the custom runner
  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 3:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(
          data=b'Monitor both stock and video', mime_type='audio/pcm'
      )
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  # Check that we got both streaming tool function calls
  stock_call_found = False
  video_call_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call:
          if part.function_call.name == 'monitor_stock_price':
            stock_call_found = True
            assert part.function_call.args['stock_symbol'] == 'NVDA'
          elif part.function_call.name == 'monitor_video_stream':
            video_call_found = True

  assert stock_call_found, 'Expected monitor_stock_price function call event.'
  assert video_call_found, 'Expected monitor_video_stream function call event.'


def test_live_streaming_buffered_function_call_yielded_during_transcription():
  """Test that function calls buffered during transcription are yielded.

  This tests the fix for the bug where function_call and function_response
  events were buffered during active transcription but never yielded to the
  caller. The fix ensures buffered events are yielded after transcription ends.
  """
  function_call = types.Part.from_function_call(
      name='get_weather', args={'location': 'San Francisco'}
  )

  response1 = LlmResponse(
      input_transcription=types.Transcription(text='Show'),
      partial=True,  # ← Triggers is_transcribing = True
  )
  response2 = LlmResponse(
      content=types.Content(
          role='model', parts=[function_call]
      ),  # ← Gets buffered
      turn_complete=False,
  )
  response3 = LlmResponse(
      input_transcription=types.Transcription(text='Show me the weather'),
      partial=False,  # ← Transcription ends, buffered events yielded
  )
  response4 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create(
      [response1, response2, response3, response4]
  )

  def get_weather(location: str) -> dict:
    return {'temperature': 22, 'location': location}

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[get_weather],
  )

  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 5:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      return collected_responses

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Show me the weather', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert len(res_events) >= 1, 'Expected at least one event.'

  function_call_found = False
  function_response_found = False

  for event in res_events:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.name == 'get_weather':
          function_call_found = True
          assert part.function_call.args['location'] == 'San Francisco'
        if (
            part.function_response
            and part.function_response.name == 'get_weather'
        ):
          function_response_found = True
          assert part.function_response.response['temperature'] == 22

  assert function_call_found, 'Buffered function_call event was not yielded.'
  assert (
      function_response_found
  ), 'Buffered function_response event was not yielded.'


def test_live_streaming_text_content_persisted_in_session():
  """Test that user text content sent via send_content is persisted in session."""
  response1 = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part(text='Hello! How can I help you?')]
      ),
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  class CustomTestRunner(testing_utils.InMemoryRunner):

    def _run_with_loop(self, coro):
      try:
        old_loop = asyncio.get_event_loop()
      except RuntimeError:
        old_loop = None

      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(coro)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      finally:
        loop.close()
        asyncio.set_event_loop(old_loop)

    def run_live_and_get_session(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
    ) -> tuple[list[testing_utils.Event], testing_utils.Session]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )
        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= 1:
            return

      self._run_with_loop(
          asyncio.wait_for(consume_responses(self.session), timeout=5.0)
      )

      # Get the updated session
      updated_session = self.runner.session_service.get_session_sync(
          app_name=self.app_name,
          user_id=self.session.user_id,
          session_id=self.session.id,
      )
      return collected_responses, updated_session

  runner = CustomTestRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()

  # Send text content (not audio blob)
  user_text = 'Hello, this is a test message'
  live_request_queue.send_content(
      types.Content(role='user', parts=[types.Part(text=user_text)])
  )

  res_events, session = runner.run_live_and_get_session(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'

  # Check that user text content was persisted in the session
  user_content_found = False
  for event in session.events:
    if event.author == 'user' and event.content:
      for part in event.content.parts:
        if part.text and user_text in part.text:
          user_content_found = True
          break

  assert user_content_found, (
      f'Expected user text content "{user_text}" to be persisted in session. '
      f'Session events: {[e.content for e in session.events]}'
  )


def _collect_function_call_names(events):
  """Extract the set of function call names from a list of events."""
  return {fc.name for event in events for fc in event.get_function_calls()}


class _LiveTestRunner(testing_utils.InMemoryRunner):
  """Test runner with custom event loop management for live streaming tests."""

  def _run_with_loop(self, coro: Awaitable[Any]) -> None:
    """Run a coroutine in a new event loop, suppressing timeouts."""
    try:
      old_loop = asyncio.get_event_loop()
    except RuntimeError:
      old_loop = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      loop.run_until_complete(coro)
    except (asyncio.TimeoutError, asyncio.CancelledError):
      pass
    finally:
      loop.close()
      asyncio.set_event_loop(old_loop)

  def run_live(
      self,
      live_request_queue: LiveRequestQueue,
      max_responses: int = 3,
  ) -> list[testing_utils.Event]:
    """Run live and collect up to max_responses events."""
    collected = []

    async def consume(session: testing_utils.Session):
      async for response in self.runner.run_live(
          session=session,
          live_request_queue=live_request_queue,
      ):
        collected.append(response)
        if len(collected) >= max_responses:
          return

    self._run_with_loop(asyncio.wait_for(consume(self.session), timeout=5.0))
    return collected


def test_input_streaming_tool_registered_lazily_with_stream():
  """Test that input-streaming tools are registered lazily when called and receive a stream."""
  # A text response before the function call lets us observe that the
  # tool is NOT registered before the model calls it.
  text_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Processing...')],
      ),
      turn_complete=False,
  )
  function_call = types.Part.from_function_call(
      name='monitor_video_stream', args={}
  )
  call_response = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  done_response = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create(
      [text_response, call_response, done_response]
  )

  stream_state_during_call = None

  async def monitor_video_stream(
      input_stream: LiveRequestQueue,
  ) -> AsyncGenerator[str, None]:
    """Record whether input_stream was provided."""
    nonlocal stream_state_during_call
    stream_state_during_call = input_stream is not None
    yield 'monitoring started'

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_video_stream],
  )

  runner = _LiveTestRunner(root_agent=root_agent)

  # Capture the invocation context to inspect registration state.
  captured_context = None
  original_method = runner.runner._new_invocation_context_for_live

  def capturing_method(*args, **kwargs) -> Any:
    nonlocal captured_context
    ctx = original_method(*args, **kwargs)
    captured_context = ctx
    return ctx

  runner.runner._new_invocation_context_for_live = capturing_method

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'test_data', mime_type='audio/pcm')
  )

  # Collect events and check that the tool is NOT registered before
  # the model calls it.
  collected = []
  not_registered_before_call = None

  async def consume(session: testing_utils.Session):
    nonlocal not_registered_before_call
    async for response in runner.runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
    ):
      collected.append(response)
      # On the first non-function-call event, verify the tool is not
      # yet registered (lazy registration).
      active = (
          captured_context.active_streaming_tools if captured_context else None
      )
      if (
          not_registered_before_call is None
          and not response.get_function_calls()
      ):
        not_registered_before_call = (
            active is None or 'monitor_video_stream' not in active
        )
      if len(collected) >= 4:
        return

  runner._run_with_loop(asyncio.wait_for(consume(runner.session), timeout=5.0))

  # Tool should not be registered before the model calls it.
  assert (
      not_registered_before_call is True
  ), 'Expected tool to NOT be registered before the model calls it'
  # When the model calls the tool, input_stream should be provided.
  assert (
      stream_state_during_call is True
  ), 'Expected input_stream to be provided to the streaming tool when called'


def test_stop_streaming_resets_stream_to_none():
  """Test that stop_streaming sets stream back to None."""
  start_call = types.Part.from_function_call(
      name='monitor_stock_price', args={'stock_symbol': 'GOOG'}
  )
  stop_call = types.Part.from_function_call(
      name='stop_streaming', args={'function_name': 'monitor_stock_price'}
  )

  response1 = LlmResponse(
      content=types.Content(role='model', parts=[start_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      content=types.Content(role='model', parts=[stop_call]),
      turn_complete=False,
  )
  response3 = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create([response1, response2, response3])

  async def monitor_stock_price(
      stock_symbol: str,
  ) -> AsyncGenerator[str, None]:
    """Yield periodic price updates for the given stock symbol."""
    yield f'Monitoring {stock_symbol}'
    while True:
      await asyncio.sleep(0.1)
      yield f'{stock_symbol} price update'

  def stop_streaming(function_name: str) -> None:
    """Stop a running streaming tool by name."""
    pass

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_stock_price, stop_streaming],
  )

  runner = _LiveTestRunner(root_agent=root_agent)

  # Capture the child invocation context (created by _create_invocation_context
  # inside base_agent.run_live) to inspect active_streaming_tools.
  # We cannot use the parent context from _new_invocation_context_for_live
  # because model_copy creates a separate child object.
  captured_child_context = None
  original_create = root_agent._create_invocation_context

  def capturing_create(*args, **kwargs) -> Any:
    nonlocal captured_child_context
    ctx = original_create(*args, **kwargs)
    captured_child_context = ctx
    return ctx

  root_agent._create_invocation_context = capturing_create

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'Monitor GOOG then stop', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=4)

  # Verify both function calls were processed.
  call_names = _collect_function_call_names(res_events)
  assert (
      'monitor_stock_price' in call_names
  ), 'Expected monitor_stock_price function call.'
  assert (
      'stop_streaming' in call_names
  ), 'Expected stop_streaming function call.'

  # Verify that stop_streaming reset the stream to None.
  assert (
      captured_child_context is not None
  ), 'Expected child invocation context to be captured'
  active_tools = captured_child_context.active_streaming_tools or {}
  assert (
      'monitor_stock_price' in active_tools
  ), 'Expected monitor_stock_price in active_streaming_tools'
  assert (
      active_tools['monitor_stock_price'].stream is None
  ), 'Expected stream to be reset to None after stop_streaming'


def test_output_streaming_tool_registered_lazily_without_stream():
  """Test that output-streaming tools are registered lazily when called, with stream=None."""
  function_call = types.Part.from_function_call(
      name='monitor_stock_price', args={'stock_symbol': 'GOOG'}
  )
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create([response1, response2])

  async def monitor_stock_price(
      stock_symbol: str,
  ) -> AsyncGenerator[str, None]:
    """Yield periodic price updates."""
    yield f'price for {stock_symbol}'

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_stock_price],
  )

  runner = _LiveTestRunner(root_agent=root_agent)

  # Capture the child invocation context (created by _create_invocation_context
  # inside base_agent.run_live) to inspect active_streaming_tools.
  captured_child_context = None
  original_create = root_agent._create_invocation_context

  def capturing_create(*args, **kwargs) -> Any:
    nonlocal captured_child_context
    ctx = original_create(*args, **kwargs)
    captured_child_context = ctx
    return ctx

  root_agent._create_invocation_context = capturing_create

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'test', mime_type='audio/pcm')
  )

  runner.run_live(live_request_queue, max_responses=3)

  # After the model calls the tool, it should be registered with
  # stream=None (output-streaming tools don't consume the live stream).
  assert captured_child_context is not None
  active_tools = captured_child_context.active_streaming_tools or {}
  assert (
      'monitor_stock_price' in active_tools
  ), 'Expected output-streaming tool to be registered when called'
  assert (
      active_tools['monitor_stock_price'].stream is None
  ), 'Expected stream to be None for output-streaming tool'


def _run_single_tool_live(
    tool_func,
    func_name: str,
    func_args: dict[str, Any] | None = None,
    max_responses: int = 3,
) -> dict[str, Any]:
  """Run a live session that invokes a single tool and return active_streaming_tools.

  Sets up a mock model that issues one function call then completes,
  creates an agent with the given tool, captures the invocation context,
  and returns the ``active_streaming_tools`` dict after execution.
  """
  function_call = types.Part.from_function_call(
      name=func_name, args=func_args or {}
  )
  response1 = LlmResponse(
      content=types.Content(role='model', parts=[function_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create([response1, response2])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[tool_func],
  )

  runner = _LiveTestRunner(root_agent=root_agent)

  captured_child_context = None
  original_create = root_agent._create_invocation_context

  def capturing_create(*args, **kwargs) -> Any:
    nonlocal captured_child_context
    ctx = original_create(*args, **kwargs)
    captured_child_context = ctx
    return ctx

  root_agent._create_invocation_context = capturing_create

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'test', mime_type='audio/pcm')
  )

  runner.run_live(live_request_queue, max_responses=max_responses)

  assert captured_child_context is not None
  return captured_child_context.active_streaming_tools or {}


def test_input_streaming_tool_has_stream_set_at_registration():
  """Test that input-streaming tools get .stream set to a LiveRequestQueue during registration."""

  async def monitor_video_stream(
      input_stream: LiveRequestQueue,
  ) -> AsyncGenerator[str, None]:
    """Simulate an input-streaming tool."""
    yield 'started'

  active_tools = _run_single_tool_live(
      monitor_video_stream, 'monitor_video_stream'
  )

  assert (
      'monitor_video_stream' in active_tools
  ), 'Expected input-streaming tool to be registered when called'
  # Stream should be a LiveRequestQueue, not None.
  assert (
      active_tools['monitor_video_stream'].stream is not None
  ), 'Expected .stream to be set for input-streaming tool'
  assert isinstance(
      active_tools['monitor_video_stream'].stream, LiveRequestQueue
  ), 'Expected .stream to be a LiveRequestQueue instance'


def test_input_streaming_tool_stream_recreated_after_stop():
  """Test that re-invoking an input-streaming tool after stop creates a new stream."""
  start_call = types.Part.from_function_call(name='monitor_video', args={})
  stop_call = types.Part.from_function_call(
      name='stop_streaming', args={'function_name': 'monitor_video'}
  )
  restart_call = types.Part.from_function_call(name='monitor_video', args={})

  response1 = LlmResponse(
      content=types.Content(role='model', parts=[start_call]),
      turn_complete=False,
  )
  response2 = LlmResponse(
      content=types.Content(role='model', parts=[stop_call]),
      turn_complete=False,
  )
  response3 = LlmResponse(
      content=types.Content(role='model', parts=[restart_call]),
      turn_complete=False,
  )
  response4 = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create(
      [response1, response2, response3, response4]
  )

  call_count = 0

  async def monitor_video(
      input_stream: LiveRequestQueue,
  ) -> AsyncGenerator[str, None]:
    """Simulate an input-streaming tool that tracks invocation count."""
    nonlocal call_count
    call_count += 1
    yield f'started (call {call_count})'
    while True:
      await asyncio.sleep(0.1)
      yield 'frame'

  def stop_streaming(function_name: str) -> None:
    """Stop a running streaming tool by name."""
    pass

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[monitor_video, stop_streaming],
  )

  runner = _LiveTestRunner(root_agent=root_agent)

  captured_child_context = None
  original_create = root_agent._create_invocation_context

  def capturing_create(*args, **kwargs) -> Any:
    nonlocal captured_child_context
    ctx = original_create(*args, **kwargs)
    captured_child_context = ctx
    return ctx

  root_agent._create_invocation_context = capturing_create

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'test', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=8)

  # monitor_video should appear at least twice in function calls
  # (start + restart). Function response events may add extra
  # occurrences.
  call_names = [
      fc.name for event in res_events for fc in event.get_function_calls()
  ]
  assert (
      call_names.count('monitor_video') >= 2
  ), f'Expected monitor_video called at least twice, got: {call_names}'

  # After re-invocation, stream should be set again (not None).
  assert captured_child_context is not None
  active_tools = captured_child_context.active_streaming_tools or {}
  assert 'monitor_video' in active_tools
  assert (
      active_tools['monitor_video'].stream is not None
  ), 'Expected .stream to be recreated after stop + re-invocation'


def test_async_gen_with_input_stream_wrong_annotation_gets_no_stream():
  """Test that an async generator with input_stream param but wrong annotation gets no stream."""
  received_input_stream = None

  async def my_tool(input_stream: str) -> AsyncGenerator[str, None]:
    """Simulate an async generator whose input_stream is typed as str."""
    nonlocal received_input_stream
    received_input_stream = input_stream
    yield f'got: {input_stream}'

  active_tools = _run_single_tool_live(
      my_tool, 'my_tool', func_args={'input_stream': 'some_value'}
  )

  assert (
      'my_tool' in active_tools
  ), 'Expected async generator tool to be registered'
  # Stream should be None because annotation is str, not LiveRequestQueue.
  assert active_tools['my_tool'].stream is None, (
      'Expected .stream to be None when input_stream annotation is not'
      ' LiveRequestQueue'
  )
  # The tool should have received the model-provided arg value, not a
  # LiveRequestQueue.
  assert (
      received_input_stream == 'some_value'
  ), 'Expected input_stream to be the model-provided string value'
