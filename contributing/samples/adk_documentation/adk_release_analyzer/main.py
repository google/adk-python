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
import logging
import time

from adk_documentation.adk_release_analyzer import agent
from adk_documentation.settings import CODE_OWNER
from adk_documentation.settings import CODE_REPO
from adk_documentation.settings import DOC_OWNER
from adk_documentation.settings import DOC_REPO
from adk_documentation.utils import call_agent_async
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "adk_release_analyzer"
USER_ID = "adk_release_analyzer_user"

logs.setup_adk_logger(level=logging.DEBUG)


async def main():
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )
  session = await runner.session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
  )

  response = await call_agent_async(
      runner,
      USER_ID,
      session.id,
      "Please analyze the most recent two releases of ADK Python!",
  )
  print(f"<<<< Agent Final Output: {response}\n")


if __name__ == "__main__":
  start_time = time.time()
  print(
      f"Start analyzing {CODE_OWNER}/{CODE_REPO} releases for"
      f" {DOC_OWNER}/{DOC_REPO} updates at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
  )
  print("-" * 80)
  asyncio.run(main())
  print("-" * 80)
  end_time = time.time()
  print(
      "Triaging finished at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}",
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
