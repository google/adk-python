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
from datetime import datetime
from datetime import timedelta
from typing import cast

import agent
from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def main():
  app_name = 'my_app'
  user_id_1 = 'user1'
  runner = InMemoryRunner(
      app_name=app_name,
      agent=agent.root_agent,
  )

  async def run_prompt(session: Session, new_message: str) -> Session:
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print('** User says:', content.model_dump(exclude_none=True))
    async for event in runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
    ):
      if not event.content or not event.content.parts:
        continue
      if event.content.parts[0].text:
        print(f'** {event.author}: {event.content.parts[0].text}')
      elif event.content.parts[0].function_call:
        print(
            f'** {event.author}: fc /'
            f' {event.content.parts[0].function_call.name} /'
            f' {event.content.parts[0].function_call.args}\n'
        )
      elif event.content.parts[0].function_response:
        print(
            f'** {event.author}: fr /'
            f' {event.content.parts[0].function_response.name} /'
            f' {event.content.parts[0].function_response.response}\n'
        )

    return cast(
        Session,
        await runner.session_service.get_session(
            app_name=app_name, user_id=user_id_1, session_id=session.id
        ),
    )

  session_1 = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )

  print(f'----Session to create memory: {session_1.id} ----------------------')
  session_1 = await run_prompt(session_1, 'Hi')
  session_1 = await run_prompt(session_1, 'My name is Jack')
  session_1 = await run_prompt(session_1, 'I like badminton.')
  session_1 = await run_prompt(
      session_1,
      f'I ate a burger on {(datetime.now() - timedelta(days=1)).date()}.',
  )
  session_1 = await run_prompt(
      session_1,
      f'I ate a banana on {(datetime.now() - timedelta(days=2)).date()}.',
  )
  print('Saving session to memory service...')
  if runner.memory_service:
    await runner.memory_service.add_session_to_memory(session_1)
  print('-------------------------------------------------------------------')

  session_2 = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )
  print(f'----Session to use memory: {session_2.id} ----------------------')
  session_2 = await run_prompt(session_2, 'Hi')
  session_2 = await run_prompt(session_2, 'What do I like to do?')
  # ** memory_agent: You like badminton.
  session_2 = await run_prompt(session_2, 'When did I say that?')
  # ** memory_agent: You said you liked badminton on ...
  session_2 = await run_prompt(session_2, 'What did I eat yesterday?')
  # ** memory_agent: You ate a burger yesterday...
  print('-------------------------------------------------------------------')


if __name__ == '__main__':
  asyncio.run(main())
