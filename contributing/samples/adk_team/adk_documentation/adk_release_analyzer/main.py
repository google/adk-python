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

import argparse
import asyncio
import logging
import os
import time

from adk_documentation.adk_release_analyzer import agent
from adk_documentation.settings import CODE_OWNER
from adk_documentation.settings import CODE_REPO
from adk_documentation.settings import DOC_OWNER
from adk_documentation.settings import DOC_REPO
from adk_documentation.utils import call_agent_async
from google.adk.cli.utils import logs
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

APP_NAME = "adk_release_analyzer"
USER_ID = "adk_release_analyzer_user"
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

logs.setup_adk_logger(level=logging.INFO)


async def main():
  parser = argparse.ArgumentParser(description="ADK Release Analyzer")
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Resume from the last session instead of starting fresh.",
  )
  parser.add_argument(
      "--start-tag",
      type=str,
      default=None,
      help="The older release tag (base) for comparison, e.g. v1.26.0.",
  )
  parser.add_argument(
      "--end-tag",
      type=str,
      default=None,
      help="The newer release tag (head) for comparison, e.g. v1.27.0.",
  )
  args = parser.parse_args()

  session_service = DatabaseSessionService(db_url=DB_URL)

  if args.resume:
    # Find the most recent session to resume
    sessions_response = await session_service.list_sessions(
        app_name=APP_NAME, user_id=USER_ID
    )
    if not sessions_response.sessions:
      print("No previous session found. Starting fresh.")
      args.resume = False

  if args.resume:
    # Resume: use existing session with resume_pipeline (skip planner)
    last_session = sessions_response.sessions[-1]
    session_id = last_session.id
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    state = session.state
    group_index = state.get("current_group_index", 0)
    total_groups = len(state.get("file_groups", []))
    num_recs = len(state.get("recommendations", []))
    print(f"Resuming session {session_id}")
    print(
        f"  Progress: group {group_index + 1}/{total_groups},"
        f" {num_recs} recommendations so far"
    )
    print(
        f"  Release: {state.get('start_tag', '?')} →"
        f" {state.get('end_tag', '?')}"
    )

    runner = Runner(
        agent=agent.resume_pipeline,
        app_name=APP_NAME,
        session_service=session_service,
    )
    prompt = "Resume analyzing the remaining file groups."
  else:
    # Fresh run
    runner = Runner(
        agent=agent.root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )
    session_id = session.id
    if args.start_tag and args.end_tag:
      prompt = (
          f"Please analyze ADK Python releases from {args.start_tag} to"
          f" {args.end_tag}!"
      )
    elif args.end_tag:
      prompt = (
          f"Please analyze the ADK Python release {args.end_tag} against its"
          " previous release!"
      )
    else:
      prompt = "Please analyze the most recent two releases of ADK Python!"

  print(f"Session ID: {session_id}")
  print("-" * 80)

  response = await call_agent_async(runner, USER_ID, session_id, prompt)
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
