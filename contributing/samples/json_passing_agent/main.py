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
import time

import agent
from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def main():
  """Runs the pizza ordering agent."""
  app_name = 'pizza_app'
  user_id = 'user1'
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
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content and event.content.parts and event.content.parts[0].text:
        print(f'** {event.author}: {event.content.parts[0].text}')

  start_time = time.time()
  print('Start time:', time.ctime(start_time))
  print('------------------------------------')
  await run_prompt(
      session,
      "I'd like a large pizza with pepperoni and mushrooms on a thin crust.",
  )
  print('------------------------------------')
  end_time = time.time()
  print('End time:', time.ctime(end_time))
  print(f'Total time: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
  asyncio.run(main())
