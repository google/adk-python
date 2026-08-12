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
import time

from adk_documentation.adk_docs_updater import agent
from adk_documentation.settings import CODE_OWNER
from adk_documentation.settings import CODE_REPO
from adk_documentation.settings import DOC_OWNER
from adk_documentation.settings import DOC_REPO
from adk_documentation.tools import get_issue
from adk_documentation.utils import call_agent_async
from adk_documentation.utils import parse_suggestions
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "adk_docs_updater"
USER_ID = "adk_docs_updater_user"

logs.setup_adk_logger(level=logging.INFO)


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
  issue_title = issue.get("title", "")
  issue_body = issue.get("body", "")

  # Parse numbered suggestions from issue body
  suggestions = parse_suggestions(issue_body)

  if not suggestions:
    print(f"No numbered suggestions found in issue #{issue_number}.")
    print("Falling back to processing the entire issue as a single task.")
    suggestions = [(1, issue_body)]

  print(f"Found {len(suggestions)} suggestion(s) in issue #{issue_number}.")
  print("=" * 80)

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  results = []
  for suggestion_num, suggestion_text in suggestions:
    print(f"\n>>> Processing suggestion #{suggestion_num}...")
    print("-" * 80)

    # Create a new session for each suggestion to avoid context interference
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    prompt = f"""
      Please update the ADK docs according to suggestion #{suggestion_num} from issue #{issue_number}.

      Issue title: {issue_title}

      Suggestion to process:
      {suggestion_text}

      Note: Focus only on this specific suggestion. Create exactly one pull request for this suggestion.
    """

    try:
      response = await call_agent_async(
          runner,
          USER_ID,
          session.id,
          prompt,
      )
      results.append({
          "suggestion_num": suggestion_num,
          "status": "success",
          "response": response,
      })
      print(f"<<<< Suggestion #{suggestion_num} completed.")
    except Exception as e:
      results.append({
          "suggestion_num": suggestion_num,
          "status": "error",
          "error": str(e),
      })
      print(f"<<<< Suggestion #{suggestion_num} failed: {e}")

    print("-" * 80)

  # Print summary
  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  successful = [r for r in results if r["status"] == "success"]
  failed = [r for r in results if r["status"] == "error"]
  print(
      f"Total: {len(results)}, Success: {len(successful)}, Failed:"
      f" {len(failed)}"
  )
  if failed:
    print("\nFailed suggestions:")
    for r in failed:
      print(f"  - Suggestion #{r['suggestion_num']}: {r['error']}")


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
