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

from adk_triaging_agent import agent
from adk_triaging_agent.settings import EVENT_NAME, ISSUE_COUNT_TO_PROCESS, ISSUE_NUMBER, OWNER, REPO
from adk_triaging_agent.utils import parse_number_string
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner, Runner
from google.genai import types

APP_NAME = "adk_triage_app"
USER_ID = "adk_triage_user"


async def call_agent_async(
    runner: Runner, user_id: str, session_id: str, prompt: str
) -> str:
  """Call the agent asynchronously with the user's prompt."""
  content = types.Content(
      role="user", parts=[types.Part.from_text(text=prompt)]
  )

  final_response_text = ""
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(save_input_blobs_as_artifacts=False),
  ):
    if (
        event.content
        and event.content.parts
        and hasattr(event.content.parts[0], "text")
        and event.content.parts[0].text
    ):
      print(f"** {event.author} (ADK): {event.content.parts[0].text}")
      if event.author == agent.root_agent.name:
        final_response_text += event.content.parts[0].text

  return final_response_text


async def main_sync():
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )
  session = await runner.session_service.create_session(
      user_id=USER_ID,
      app_name=APP_NAME,
  )

  if EVENT_NAME == "issues" and ISSUE_NUMBER:
    print(f"EVENT: Processing specific issue due to '{EVENT_NAME}' event.")
    issume_number = parse_number_string(ISSUE_NUMBER)
    if not issume_number:
      print(f"Error: Invalid issue number received: {ISSUE_NUMBER}.")
      return
    prompt = (
        f"Please triage issue #{issume_number}. If the issue is already"
        " labelled, please don't change it, just return the labels. If the"
        " issue is not labelled, please label it. Only label it, do not"
        " process any other issues."
    )
  else:
    print(f"EVENT: Processing batch of issues (event: {EVENT_NAME}).")
    issue_count = parse_number_string(ISSUE_COUNT_TO_PROCESS, default_value=3)
    prompt = f"Please triage the most recent {issue_count} issues."

  response = await call_agent_async(runner, USER_ID, session.id, prompt)
  print(f"<<<< Agent Final Output: {response}\n")


if __name__ == "__main__":
  start_time = time.time()
  print(
      f"Start traging {OWNER}/{REPO} issues at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
  )
  print("-" * 80)
  asyncio.run(main_sync())
  print("-" * 80)
  end_time = time.time()
  print(
      "Traging finished at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}",
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
