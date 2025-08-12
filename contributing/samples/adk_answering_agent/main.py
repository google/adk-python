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
import logging
import time

from adk_answering_agent import agent
from adk_answering_agent.settings import DISCUSSION_NUMBER
from adk_answering_agent.settings import OWNER
from adk_answering_agent.settings import REPO
from adk_answering_agent.utils import call_agent_async
from adk_answering_agent.utils import parse_number_string
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "adk_answering_app"
USER_ID = "adk_answering_user"

logs.setup_adk_logger(level=logging.DEBUG)


async def main():
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  discussion_number = parse_number_string(DISCUSSION_NUMBER)
  if not discussion_number:
    print(f"Error: Invalid discussion number received: {DISCUSSION_NUMBER}.")
    return

  prompt = (
      f"Please check discussion #{discussion_number} see if you can help answer"
      " the question or provide some information!"
  )
  response = await call_agent_async(runner, USER_ID, session.id, prompt)
  print(f"<<<< Agent Final Output: {response}\n")


if __name__ == "__main__":
  start_time = time.time()
  print(
      f"Start Q&A checking on {OWNER}/{REPO} discussion #{DISCUSSION_NUMBER} at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
  )
  print("-" * 80)
  asyncio.run(main())
  print("-" * 80)
  end_time = time.time()
  print(
      "Q&A checking finished at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}",
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
