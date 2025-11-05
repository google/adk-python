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

import argparse
import asyncio
import logging
import time

from adk_documentation.adk_docs_updater import agent
from adk_documentation.settings import CODE_OWNER
from adk_documentation.settings import CODE_REPO
from adk_documentation.settings import DOC_OWNER
from adk_documentation.settings import DOC_REPO
from adk_documentation.tools import get_issue
from adk_documentation.utils import call_agent_async
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "adk_docs_updater"
USER_ID = "adk_docs_updater_user"

logs.setup_adk_logger(level=logging.DEBUG)


def process_arguments():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="A script that creates pull requests to update ADK docs.",
      epilog=(
          "Example usage: \n"
          "\tpython -m adk_docs_updater.main --issue_number 123\n"
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  group = parser.add_mutually_exclusive_group(required=True)

  group.add_argument(
      "--issue_number",
      type=int,
      metavar="NUM",
      help="Answer a specific issue number.",
  )

  return parser.parse_args()


async def main():
  args = process_arguments()
  if not args.issue_number:
    print("Please specify an issue number using --issue_number flag")
    return
  issue_number = args.issue_number

  get_issue_response = get_issue(DOC_OWNER, DOC_REPO, issue_number)
  if get_issue_response["status"] != "success":
    print(f"Failed to get issue {issue_number}: {get_issue_response}\n")
    return
  issue = get_issue_response["issue"]

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
      f"Please update the ADK docs according to the following issue:\n{issue}",
  )
  print(f"<<<< Agent Final Output: {response}\n")


if __name__ == "__main__":
  start_time = time.time()
  print(
      f"Start creating pull requests to update {DOC_OWNER}/{DOC_REPO} docs"
      f" according the {CODE_OWNER}/{CODE_REPO} at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
  )
  print("-" * 80)
  asyncio.run(main())
  print("-" * 80)
  end_time = time.time()
  print(
      "Updating finished at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}",
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
