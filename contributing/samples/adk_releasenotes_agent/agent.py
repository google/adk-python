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

import base64
from datetime import datetime
from datetime import timedelta
import os
from typing import Any

from google.adk import Agent
import requests


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
CHANGELOG_FILE = "CHANGELOG.md"


def get_commits(days: int = 7) -> list[dict[str, Any]]:
  """Lists git commits in the last a few days using the GitHub REST API.

  Args:
      days (int, optional): The number of days to get commits.

  Returns:
      list: A list of dictionaries, where each dictionary represents a commit.
            Returns an empty list if there's an error or no commits found.
  """

  # Calculate the since date.
  since_date = (datetime.now() - timedelta(days=days)).isoformat() + "Z"

  url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"

  headers = {
      "Accept": "application/vnd.github+json",
      "Authorization": f"Bearer {GITHUB_TOKEN}",
      "X-GitHub-Api-Version": "2022-11-28",
  }

  params = {"since": since_date, "per_page": 100}

  all_commits = []
  page = 1
  while True:
    params["page"] = page
    try:
      response = requests.get(url, headers=headers, params=params)
      response.raise_for_status()

      commits_on_page = response.json()
      if not commits_on_page:
        break

      all_commits.extend(commits_on_page)

      # Check for Link header to handle pagination
      if "Link" in response.headers:
        link_header = response.headers["Link"]
        # Simple check for 'next' link to continue pagination
        if 'rel="next"' not in link_header:
          break  # No more pages
      else:
        break  # No Link header, likely only one page of results

      page += 1

    except requests.exceptions.HTTPError as e:
      print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
      if (
          e.response.status_code == 403
          and "X-RateLimit-Remaining" in e.response.headers
          and e.response.headers["X-RateLimit-Remaining"] == "0"
      ):
        print(
            "Rate limit exceeded. Please wait or use a Personal Access Token."
        )
      return []
    except requests.exceptions.ConnectionError as e:
      print(f"Connection Error: {e}")
      return []
    except requests.exceptions.Timeout as e:
      print(f"Timeout Error: {e}")
      return []
    except requests.exceptions.RequestException as e:
      print(f"An error occurred: {e}")
      return []

  return all_commits


def merge_commit_messages(commits: dict[str, Any]) -> str:
  """Returns the commit message for a given commit dictionary."""
  merged_commit_messages = []

  for commit_data in commits:
    commit_sha = commit_data["sha"]
    commit_message = (
        commit_data["commit"]["message"].strip().split("\n")[0]
    )  # First line of message
    merged_commit_messages.append(
        commit_message + f"[{commit_sha[:7]}]({commit_sha})"
    )
  return "".join(merged_commit_messages)


def get_file_content():
  """Fetches the content and SHA of a file from a GitHub repository.

  Returns:
      tuple: (content, sha) of the file, or (None, None) if not found/error.
  """
  url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{CHANGELOG_FILE}"
  headers = {
      "Authorization": f"Bearer {GITHUB_TOKEN}",
      "Accept": "application/vnd.github.v3+json",
      "X-GitHub-Api-Version": "2022-11-28",
  }

  try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    file_data = response.json()

    # The content is base64 encoded.
    content_encoded = file_data["content"]
    content_decoded = base64.b64decode(content_encoded).decode("utf-8")

    sha = file_data["sha"]
    return content_decoded, sha

  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      print(f"File '{CHANGELOG_FILE}' not found in {OWNER}/{REPO}.")
      return "", None
    else:
      print(
          f"Error fetching file: {e.response.status_code} - {e.response.text}"
      )
      return "", None
  except requests.exceptions.RequestException as e:
    print(f"Network or request error: {e}")
    return "", None


def prepare_release_note_prompt(days: int = 7) -> str:
  """Prepares the prompt for the release note generation."""
  current_content, _ = get_file_content()
  recent_commit_messages = merge_commit_messages(get_commits(days=days))
  return f"""The previous release notes are: {current_content}. We can get the latest release version from previous release notes. =======\nThe new release notes should contain changes including: {"".join(recent_commit_messages)}"""


system_prompt = """
You are a helpful assistant to prepare release notes.

For each query, you will call the tool prepare_release_note_prompt to prepare release note prompts.

The release notes can be summarized as below:
## Release Version (Release Date)
### Features
### Bug Fixes
### Improvements
where each section has at most 10 of bullets, with highly summarized descriptions.
Each bullet should have the format as
* <description> [commit](link)
where the link must start with https://github.com/google/adk-python/commit.

The inputs are a list of messages, which starts with `chore`, `docs`, `feat`, `fix`, `test`, `refactor` etc.
`feat` stands for a new feature.
`fix` stands for a bug fix.
`chore`, `docs`, `test`, and `refactor` stand for improvements.

For release version, we always automatically generate the minor version updates.
If the current version is 1.2.0, then next version is 1.2.1.
If the current version is 1.2.10, then next version is 1.2.11.

=================Release note example starts.
## 1.3.0 (2025-06-11)

### Features

* Add memory_service option to CLI ([416dc6f](https://github.com/google/adk-python/commit/416dc6feed26e55586d28f8c5132b31413834c88))
* Add support for display_name and description when deploying to agent engine ([aaf1f9b](https://github.com/google/adk-python/commit/aaf1f9b930d12657bfc9b9d0abd8e2248c1fc469))
* Dev UI: Trace View
  * New trace tab which contains all traces grouped by user messages
  * Click each row will open corresponding event details
  * Hover each row will highlight the corresponding message in dialog
* Dev UI: Evaluation
  * Evaluation Configuration: users can now configure custom threshold for the metrics used for each eval run ([d1b0587](https://github.com/google/adk-python/commit/d1b058707eed72fd4987d8ec8f3b47941a9f7d64))
  * Each eval case added can now be viewed and edited. Right now we only support edit of text.
  * Show the used metric in evaluation history ([6ed6351](https://github.com/google/adk-python/commit/6ed635190c86d5b2ba0409064cf7bcd797fd08da))
* Tool enhancements:
  * Add url_context_tool ([fe1de7b](https://github.com/google/adk-python/commit/fe1de7b10326a38e0d5943d7002ac7889c161826))
  * Support to customize timeout for mcpstdio connections ([54367dc](https://github.com/google/adk-python/commit/54367dcc567a2b00e80368ea753a4fc0550e5b57))
  * Introduce write protected mode to BigQuery tools ([6c999ca](https://github.com/google/adk-python/commit/6c999caa41dca3a6ec146ea42b0a794b14238ec2))



### Bug Fixes

* Agent Engine deployment:
  * Correct help text formatting for `adk deploy agent_engine` ([13f98c3](https://github.com/google/adk-python/commit/13f98c396a2fa21747e455bb5eed503a553b5b22))
  * Handle project and location in the .env properly when deploying to Agent Engine ([0c40542](https://github.com/google/adk-python/commit/0c4054200fd50041f0dce4b1c8e56292b99a8ea8))
* Fix broken agent graphs ([3b1f2ae](https://github.com/google/adk-python/commit/3b1f2ae9bfdb632b52e6460fc5b7c9e04748bd50))
* Forward `__annotations__` to the fake func for FunctionTool inspection ([9abb841](https://github.com/google/adk-python/commit/9abb8414da1055ab2f130194b986803779cd5cc5))
* Handle the case when agent loading error doesn't have msg attribute in agent loader ([c224626](https://github.com/google/adk-python/commit/c224626ae189d02e5c410959b3631f6bd4d4d5c1))
* Prevent agent_graph.py throwing when workflow agent is root agent ([4b1c218](https://github.com/google/adk-python/commit/4b1c218cbe69f7fb309b5a223aa2487b7c196038))
* Remove display_name for non-Vertex file uploads ([cf5d701](https://github.com/google/adk-python/commit/cf5d7016a0a6ccf2b522df6f2d608774803b6be4))


### Improvements

* Add DeepWiki badge to README ([f38c08b](https://github.com/google/adk-python/commit/f38c08b3057b081859178d44fa2832bed46561a9))
* Update code example in tool declaration to reflect BigQuery artifact description ([3ae6ce1](https://github.com/google/adk-python/commit/3ae6ce10bc5a120c48d84045328c5d78f6eb85d4))



=================Release note example ends.

"""


root_agent = Agent(
    model="gemini-2.0-flash",
    name="github_releasenotes_agent",
    description="Generate release notes for ADK.",
    instruction=system_prompt,
    tools=[prepare_release_note_prompt],
)
