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
import time

import agent
from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def main():
  app_name = 'parallel_tool_test_app'
  user_id_1 = 'user1'
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  session_11 = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )

  async def run_prompt(session: Session, new_message: str):
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print('** User says:', content.model_dump(exclude_none=True))
    async for event in runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
    ):
      if event.content.parts and event.content.parts[0].text:
        print(f'** {event.author}: {event.content.parts[0].text}')

  start_time = time.time()
  print('Start time:', start_time)
  print('------------------------------------')
  await run_prompt(session_11, 'Hi')
  await run_prompt(
      session_11,
      'Roll a 6-sided die and check if 2 is a prime number at the same time.',
  )
  print('\n--- Session State After Parallel Tool Call ---')
  updated_session_1 = await runner.session_service.get_session(
      app_name=app_name, user_id=user_id_1, session_id=session_11.id
  )
  print('\n--- Session State After Parallel Tool Call ---')
  if updated_session_1 and updated_session_1.state:
    print(f'Session ID: {updated_session_1.id}')
    print('Session State:')
    for key, value in updated_session_1.state.items():
      print(f'  {key}: {value}')
  else:
    print('No session state found or session could not be retrieved.')

  end_time = time.time()
  print('------------------------------------')
  print('End time:', end_time)
  print('Total time:', end_time - start_time)


if __name__ == '__main__':
  asyncio.run(main())
