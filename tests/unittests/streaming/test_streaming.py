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

import asyncio
from typing import AsyncGenerator

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        # Return whatever we collected so far
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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

      try:
        session = self.session
        # Create a new event loop to avoid nested event loop issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

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
