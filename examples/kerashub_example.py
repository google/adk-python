"""Example script demonstrating KerasHub model integration with ADK.

This script tests the KerasHub LLM integration by:
1. Loading a Gemma 3 model from KerasHub
2. Running a simple text generation via the ADK agent framework
3. Running a tool-calling example to verify function calling support

Prerequisites:
  pip install keras keras-hub
  export KAGGLE_USERNAME=<>
  export KAGGLE_KEY=<>
  export KERAS_BACKEND=jax  # or tensorflow or torch
"""

import asyncio
import os

# Set Kaggle credentials for model access
os.environ['KAGGLE_USERNAME'] = 'divyasss'
os.environ['KAGGLE_KEY'] = '5530b7417df9081efa79b26f6ed713fb'

# Set Keras backend before any keras imports
os.environ['KERAS_BACKEND'] = 'jax'


async def test_basic_generation():
  """Test basic text generation with KerasHub model."""
  from google.adk.models.kerashub_llm import KerasHubLlm
  from google.adk.models.llm_request import LlmRequest
  from google.genai import types

  print('=' * 60)
  print('Test 1: Basic Text Generation')
  print('=' * 60)

  model = KerasHubLlm(
      model='kerashub://gemma3_instruct_1b',
      keras_backend='jax',
      dtype='bfloat16',
      max_length=256,
  )

  request = LlmRequest(
      model=model.model,
      contents=[
          types.Content(
              role='user',
              parts=[types.Part.from_text(text='What is Python programming language? Answer in 2 sentences.')],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction='You are a helpful assistant. Be concise.',
      ),
  )

  print(f'Model: {model}')
  print(f'Prompt: "What is Python programming language?"')
  print('-' * 40)

  async for response in model.generate_content_async(request):
    if response.error_code:
      print(f'Error: {response.error_code} - {response.error_message}')
    elif response.content and response.content.parts:
      for part in response.content.parts:
        if part.text:
          print(f'Response: {part.text}')

  print()


async def test_with_agent():
  """Test KerasHub model with the ADK Agent framework."""
  from google.adk.agents import Agent
  from google.adk.models.kerashub_llm import KerasHubLlm
  from google.adk.runners import Runner
  from google.adk.sessions import InMemorySessionService

  print('=' * 60)
  print('Test 2: ADK Agent with KerasHub Model')
  print('=' * 60)

  model = KerasHubLlm(
      model='kerashub://gemma3_instruct_1b',
      keras_backend='jax',
      dtype='bfloat16',
      max_length=256,
  )

  agent = Agent(
      name='kerashub_agent',
      model=model,
      instruction='You are a helpful coding assistant. Be concise and clear.',
  )

  session_service = InMemorySessionService()
  runner = Runner(agent=agent, app_name='kerashub_test', session_service=session_service)

  session = await session_service.create_session(
      app_name='kerashub_test', user_id='test_user'
  )

  from google.genai import types

  user_message = types.Content(
      role='user',
      parts=[types.Part.from_text(text='Write a hello world function in Python.')],
  )

  print(f'User: Write a hello world function in Python.')
  print('-' * 40)

  async for event in runner.run_async(
      session_id=session.id, user_id='test_user', new_message=user_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f'Agent: {part.text}')

  print()


async def test_model_registry():
  """Test that the model is properly registered in the LLM registry."""
  from google.adk.models.registry import LLMRegistry

  print('=' * 60)
  print('Test 3: Model Registry Resolution')
  print('=' * 60)

  try:
    model_cls = LLMRegistry.resolve('kerashub://gemma3_instruct_1b')
    print(f'Registry resolved kerashub:// to: {model_cls.__name__}')
    print('Registry integration: PASSED')
  except ValueError as e:
    print(f'Registry resolution failed: {e}')
    print('Registry integration: FAILED')

  print()


async def test_with_tools():
  """Test KerasHub model with function calling (tool use)."""
  from google.adk.models.kerashub_llm import KerasHubLlm
  from google.adk.models.llm_request import LlmRequest
  from google.genai import types

  print('=' * 60)
  print('Test 4: Function Calling Support')
  print('=' * 60)

  model = KerasHubLlm(
      model='kerashub://gemma3_instruct_1b',
      keras_backend='jax',
      dtype='bfloat16',
      max_length=256,
  )

  # Create a request with tool declarations
  request = LlmRequest(
      model=model.model,
      contents=[
          types.Content(
              role='user',
              parts=[types.Part.from_text(text='What is the weather in Tokyo?')],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction='You are a helpful assistant.',
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name='get_weather',
                          description='Get the current weather for a location.',
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  'location': types.Schema(
                                      type=types.Type.STRING,
                                      description='The city name.',
                                  ),
                              },
                              required=['location'],
                          ),
                      )
                  ]
              )
          ],
      ),
  )

  print(f'User: What is the weather in Tokyo?')
  print(f'Available tools: get_weather(location)')
  print('-' * 40)

  async for response in model.generate_content_async(request):
    if response.error_code:
      print(f'Error: {response.error_code} - {response.error_message}')
    elif response.content and response.content.parts:
      for part in response.content.parts:
        if part.text:
          print(f'Text response: {part.text}')
        elif part.function_call:
          print(f'Function call: {part.function_call.name}({part.function_call.args})')

  print()


async def main():
  print('\nKerasHub + ADK Integration Test')
  print('=' * 60)
  print()

  # Test 3 first since it doesn't require model loading
  await test_model_registry()

  # Test basic generation
  await test_basic_generation()

  # Test with agent framework
  await test_with_agent()

  # Test function calling
  await test_with_tools()

  print('All tests completed!')


if __name__ == '__main__':
  asyncio.run(main())
