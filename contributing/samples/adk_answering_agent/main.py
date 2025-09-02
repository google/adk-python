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
import os
import sys
import time

from adk_answering_agent import agent
from adk_answering_agent.settings import OWNER
from adk_answering_agent.settings import REPO
from adk_answering_agent.utils import call_agent_async
from adk_answering_agent.utils import parse_number_string
from adk_answering_agent.utils import run_graphql_query
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
import requests

APP_NAME = "adk_answering_app"
USER_ID = "adk_answering_user"

logs.setup_adk_logger(level=logging.DEBUG)


async def list_most_recent_discussions(count: int = 1) -> list[int] | None:
  """Fetches a specified number of the most recently updated discussions.

  Args:
      count: The number of discussions to retrieve. Defaults to 1.

  Returns:
      A list of discussion numbers.
  """
  print(
      f"Attempting to fetch the {count} most recently updated discussions from"
      f" {OWNER}/{REPO}..."
  )

  query = """
    query($owner: String!, $repo: String!, $count: Int!) {
      repository(owner: $owner, name: $repo) {
        discussions(
          first: $count
          orderBy: {field: UPDATED_AT, direction: DESC}
        ) {
          nodes {
            title
            number
            updatedAt
            author {
              login
            }
          }
        }
      }
    }
    """
  variables = {"owner": OWNER, "repo": REPO, "count": count}

  try:
    response = run_graphql_query(query, variables)

    if "errors" in response:
      print(f"Error from GitHub API: {response['errors']}", file=sys.stderr)
      return None

    discussions = (
        response.get("data", {})
        .get("repository", {})
        .get("discussions", {})
        .get("nodes", [])
    )
    return [d["number"] for d in discussions]

  except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}", file=sys.stderr)
    return None


def process_arguments():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="A script that answers questions for GitHub discussions.",
      epilog=(
          "Example usage: \n"
          "\tpython -m adk_answering_agent.main --recent 10\n"
          "\tpython -m adk_answering_agent.main --discussion_number 21\n"
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  group = parser.add_mutually_exclusive_group(required=True)

  group.add_argument(
      "--recent",
      type=int,
      metavar="COUNT",
      help="Answer the N most recently updated discussion numbers.",
  )

  group.add_argument(
      "--discussion_number",
      type=str,
      metavar="NUM",
      help="Answer a specific discussion number.",
  )

  return parser.parse_args()


async def main():
  args = process_arguments()
  discussion_numbers = []

  if args.recent:
    fetched_numbers = await list_most_recent_discussions(count=args.recent)
    if not fetched_numbers:
      print("No discussions found. Exiting...", file=sys.stderr)
      return
    discussion_numbers = fetched_numbers
  elif args.discussion_number:
    discussion_number = parse_number_string(args.discussion_number)
    if not discussion_number:
      print(
          "Error: Invalid discussion number received:"
          f" {args.discussion_number}."
      )
      return
    discussion_numbers = [discussion_number]

  print(f"Will try to answer discussions: {discussion_numbers}...")

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  for discussion_number in discussion_numbers:
    if len(discussion_numbers) > 1:
      print("#" * 80)
      print(f"Starting to process discussion #{discussion_number}...")
    # Create a new session for each discussion to avoid interference.
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    prompt = (
        f"Please check discussion #{discussion_number} see if you can help"
        " answer the question or provide some information!"
    )
    response = await call_agent_async(runner, USER_ID, session.id, prompt)
    print(f"<<<< Agent Final Output: {response}\n")


if __name__ == "__main__":
  start_time = time.time()
  print(
      f"Start Q&A checking on {OWNER}/{REPO} at"
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
