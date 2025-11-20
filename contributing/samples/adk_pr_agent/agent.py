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

import os

from google.adk import Agent
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")


def get_github_pr_info_http(pr_number: int) -> str | None:
  """Fetches information for a GitHub Pull Request by sending direct HTTP requests.

  Args:
      pr_number (int): The number of the Pull Request.

  Returns:
      pr_message: A string.
  """
  base_url = "https://api.github.com"

  headers = {
      "Accept": "application/vnd.github+json",
      "Authorization": f"Bearer {GITHUB_TOKEN}",
      "X-GitHub-Api-Version": "2022-11-28",
  }

  pr_message = ""

  # --- 1. Get main PR details ---
  pr_url = f"{base_url}/repos/{OWNER}/{REPO}/pulls/{pr_number}"
  print(f"Fetching PR details from: {pr_url}")
  try:
    response = requests.get(pr_url, headers=headers)
    response.raise_for_status()
    pr_data = response.json()
    pr_message += f"The PR title is: {pr_data.get('title')}\n"
  except requests.exceptions.HTTPError as e:
    print(
        f"HTTP Error fetching PR details: {e.response.status_code} - "
        f" {e.response.text}"
    )
    return None
  except requests.exceptions.RequestException as e:
    print(f"Network or request error fetching PR details: {e}")
    return None
  except Exception as e:  # pylint: disable=broad-except
    print(f"An unexpected error occurred: {e}")
    return None

  # --- 2. Fetching associated commits (paginated) ---
  commits_url = pr_data.get(
      "commits_url"
  )  # This URL is provided in the initial PR response
  if commits_url:
    print("\n--- Associated Commits in this PR: ---")
    page = 1
    while True:
      # GitHub API often uses 'per_page' and 'page' for pagination
      params = {
          "per_page": 100,
          "page": page,
      }  # Fetch up to 100 commits per page
      try:
        response = requests.get(commits_url, headers=headers, params=params)
        response.raise_for_status()
        commits_data = response.json()

        if not commits_data:  # No more commits
          break

        pr_message += "The associated commits are:\n"
        for commit in commits_data:
          message = commit.get("commit", {}).get("message", "").splitlines()[0]
          if message:
            pr_message += message + "\n"

        # Check for 'Link' header to determine if more pages exist
        # This is how GitHub's API indicates pagination
        if "Link" in response.headers:
          link_header = response.headers["Link"]
          if 'rel="next"' in link_header:
            page += 1  # Move to the next page
          else:
            break  # No more pages
        else:
          break  # No Link header, so probably only one page

      except requests.exceptions.HTTPError as e:
        print(
            f"HTTP Error fetching PR commits (page {page}):"
            f" {e.response.status_code} - {e.response.text}"
        )
        break
      except requests.exceptions.RequestException as e:
        print(
            f"Network or request error fetching PR commits (page {page}): {e}"
        )
        break
  else:
    print("Commits URL not found in PR data.")

  return pr_message


system_prompt = """
You are a helpful assistant to generate reasonable descriptions for pull requests for software engineers.

The descriptions should not be too short (e.g.: less than 3 words), or too long (e.g.: more than 30 words).

The generated description should start with `chore`, `docs`, `feat`, `fix`, `test`, or `refactor`.
`feat` stands for a new feature.
`fix` stands for a bug fix.
`chore`, `docs`, `test`, and `refactor` stand for improvements.

Some good descriptions are:
1. feat: Added implementation for `get_eval_case`, `update_eval_case` and `delete_eval_case` for the local eval sets manager.
2. feat: Provide inject_session_state as public util method.

Some bad descriptions are:
1. fix: This fixes bugs.
2. feat: This is a new feature.

"""

root_agent = Agent(
    model="gemini-2.0-flash",
    name="github_pr_agent",
    description="Generate pull request descriptions for ADK.",
    instruction=system_prompt,
)
