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
import os
import time

import agent
from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types
import requests

load_dotenv(override=True)

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

if not GITHUB_TOKEN:
  print(
      "Warning: GITHUB_TOKEN environment variable not set. API calls might"
      " fail."
  )


async def fetch_specific_issue_details(issue_number: int):
  """Fetches details for a single issue if it's unlabelled."""
  if not GITHUB_TOKEN:
    print("Cannot fetch issue details: GITHUB_TOKEN is not set.")
    return None

  url = f"{BASE_URL}/repos/{OWNER}/{REPO}/issues/{issue_number}"
  print(f"Fetching details for specific issue: {url}")
  try:
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    issue_data = response.json()
    if not issue_data.get("labels") or len(issue_data["labels"]) == 0:
      print(f"Issue #{issue_number} is unlabelled. Proceeding.")
      return {
          "number": issue_data["number"],
          "title": issue_data["title"],
          "body": issue_data.get("body", ""),
      }
    else:
      print(f"Issue #{issue_number} is already labelled. Skipping.")
      return None
  except requests.exceptions.RequestException as e:
    print(f"Error fetching issue #{issue_number}: {e}")
    if hasattr(e, "response") and e.response is not None:
      print(f"Response content: {e.response.text}")
    return None


async def main():
  app_name = "triage_app"
  user_id_1 = "triage_user"
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
    print(f"\n>>>> Agent Prompt: {prompt_text}")
    final_agent_response_parts = []
    async for event in runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
        run_config=RunConfig(save_input_blobs_as_artifacts=False),
    ):
      if event.content.parts and event.content.parts[0].text:
        print(f"** {event.author} (ADK): {event.content.parts[0].text}")
        if event.author == agent.root_agent.name:
          final_agent_response_parts.append(event.content.parts[0].text)
    print(f"<<<< Agent Final Output: {''.join(final_agent_response_parts)}\n")

  event_name = os.getenv("EVENT_NAME")
  issue_number_str = os.getenv("ISSUE_NUMBER")

  if event_name == "issues" and issue_number_str:
    print(f"EVENT: Processing specific issue due to '{event_name}' event.")
    try:
      issue_number = int(issue_number_str)
      specific_issue = await fetch_specific_issue_details(issue_number)

      if specific_issue:
        prompt = (
            f"A new GitHub issue #{specific_issue['number']} has been opened or"
            f" reopened. Title: \"{specific_issue['title']}\"\nBody:"
            f" \"{specific_issue['body']}\"\n\nBased on the rules, recommend an"
            " appropriate label and its justification."
            " Then, use the 'add_label_to_issue' tool to apply the label "
            "directly to this issue."
            f" The issue number is {specific_issue['number']}."
        )
        await run_agent_prompt(session_11, prompt)
      else:
        print(
            f"No unlabelled issue details found for #{issue_number} or an error"
            " occurred. Skipping agent interaction."
        )

    except ValueError:
      print(f"Error: Invalid ISSUE_NUMBER received: {issue_number_str}")

  else:
    print(f"EVENT: Processing batch of issues (event: {event_name}).")
    issue_count_str = os.getenv("ISSUE_COUNT_TO_PROCESS", "3")
    try:
      num_issues_to_process = int(issue_count_str)
    except ValueError:
      print(f"Warning: Invalid ISSUE_COUNT_TO_PROCESS. Defaulting to 3.")
      num_issues_to_process = 3

    prompt = (
        f"List the first {num_issues_to_process} unlabelled open issues from"
        f" the {OWNER}/{REPO} repository. For each issue, provide a summary,"
        " recommend a label with justification, and then use the"
        " 'add_label_to_issue' tool to apply the recommended label directly."
    )
    await run_agent_prompt(session_11, prompt)


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
