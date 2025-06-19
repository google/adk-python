import asyncio
import os
import time
from dotenv import load_dotenv

import agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

load_dotenv(override=True)


async def main():
  app_name = 'retry_test_app'
  user_id = 'test_user'

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )

  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  async def run_prompt(session: Session, new_message: str):
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print(f'** User says: {new_message}')

    try:
      async for event in runner.run_async(
          user_id=user_id,
          session_id=session.id,
          new_message=content,
      ):
        if event.content.parts and event.content.parts[0].text:
          print(f'** {event.author}: {event.content.parts[0].text}')
    except Exception as e:
      print(f'** Error: {e}')

  print('Testing LiteLLM with built-in retry mechanism...')
  print('Configuration:')
  print(f'- Max retries: 5')
  print(f'- Initial delay: 1.0s')
  print(f'- Max delay: 30.0s')
  print(f'- Exponential base: 2.0')
  print(f'- Jitter: True')
  print('------------------------------------')

  start_time = time.time()

  await run_prompt(
      session, 'Hello, can you confirm the retry mechanism is working?'
  )
  await run_prompt(session, 'What is 2+2?')

  end_time = time.time()
  print('------------------------------------')
  print(f'Total time: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
  if not os.getenv('GOOGLE_API_KEY'):
    print('Please set GOOGLE_API_KEY environment variable')
    exit(1)

  asyncio.run(main())
