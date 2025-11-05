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

# pylint: disable=g-importing-member

import asyncio
import time

import agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types


async def main():
  app_name = "adk_pr_app"
  user_id_1 = "adk_pr_user"
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  session_11 = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )

  async def run_agent_prompt(session: Session, prompt_text: str):
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt_text)]
    )
    final_agent_response_parts = []
    async for event in runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
        run_config=RunConfig(save_input_blobs_as_artifacts=False),
    ):
      if event.content.parts and event.content.parts[0].text:
        if event.author == agent.root_agent.name:
          final_agent_response_parts.append(event.content.parts[0].text)
    print(f"<<<< Agent Final Output: {''.join(final_agent_response_parts)}\n")

  pr_message = agent.get_github_pr_info_http(pr_number=1422)
  query = "Generate pull request description for " + pr_message
  await run_agent_prompt(session_11, query)


if __name__ == "__main__":
  start_time = time.time()
  print(
      "Script start time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
  )
  print("------------------------------------")
  asyncio.run(main())
  end_time = time.time()
  print("------------------------------------")
  print(
      "Script end time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)),
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
