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

import os
import sys
from typing import Any
from typing import Optional
from urllib.parse import urljoin

from adk_answering_agent.settings import GITHUB_GRAPHQL_URL
from adk_answering_agent.settings import GITHUB_TOKEN
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.genai import types
import requests

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def error_response(error_message: str) -> dict[str, Any]:
  return {"status": "error", "error_message": error_message}


def run_graphql_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
  """Executes a GraphQL query."""
  payload = {"query": query, "variables": variables}
  response = requests.post(
      GITHUB_GRAPHQL_URL, headers=headers, json=payload, timeout=60
  )
  response.raise_for_status()
  return response.json()


def parse_number_string(number_str: str | None, default_value: int = 0) -> int:
  """Parse a number from the given string."""
  if not number_str:
    return default_value

  try:
    return int(number_str)
  except ValueError:
    print(
        f"Warning: Invalid number string: {number_str}. Defaulting to"
        f" {default_value}.",
        file=sys.stderr,
    )
    return default_value


def _check_url_exists(url: str) -> bool:
  """Checks if a URL exists and is accessible."""
  try:
    # Set a timeout to prevent the program from waiting indefinitely.
    # allow_redirects=True ensures we correctly handle valid links
    # after redirection.
    response = requests.head(url, timeout=5, allow_redirects=True)
    # Status codes 2xx (Success) or 3xx (Redirection) are considered valid.
    return response.ok
  except requests.RequestException:
    # Catch all possible exceptions from the requests library
    # (e.g., connection errors, timeouts).
    return False


def _generate_github_url(repo_name: str, relative_path: str) -> str:
  """Generates a standard GitHub URL for a repo file."""
  return f"https://github.com/google/{repo_name}/blob/main/{relative_path}"


def convert_gcs_to_https(gcs_uri: str) -> Optional[str]:
  """Converts a GCS file link into a publicly accessible HTTPS link.

  Args:
      gcs_uri: The Google Cloud Storage link, in the format
        'gs://bucket_name/prefix/relative_path'.

  Returns:
      The converted HTTPS link as a string, or None if the input format is
      incorrect.
  """
  # Parse the GCS link
  if not gcs_uri or not gcs_uri.startswith("gs://"):
    print(f"Error: Invalid GCS link format: {gcs_uri}")
    return None

  try:
    # Strip 'gs://' and split by '/', requiring at least 3 parts
    # (bucket, prefix, path)
    parts = gcs_uri[5:].split("/", 2)
    if len(parts) < 3:
      raise ValueError(
          "GCS link must contain a bucket, prefix, and relative_path."
      )

    _, prefix, relative_path = parts
  except (ValueError, IndexError) as e:
    print(f"Error: Failed to parse GCS link '{gcs_uri}': {e}")
    return None

  # Replace .html with .md
  if relative_path.endswith(".html"):
    relative_path = relative_path.removesuffix(".html") + ".md"

  # Replace .txt with .yaml
  if relative_path.endswith(".txt"):
    relative_path = relative_path.removesuffix(".txt") + ".yaml"

  # Convert the links for adk-docs
  if prefix == "adk-docs" and relative_path.startswith("docs/"):
    path_after_docs = relative_path[len("docs/") :]
    if not path_after_docs.endswith(".md"):
      # Use the regular github url
      return _generate_github_url(prefix, relative_path)

    base_url = "https://google.github.io/adk-docs/"
    if os.path.basename(path_after_docs) == "index.md":
      # Use the directory path if it is a index file
      final_path_segment = os.path.dirname(path_after_docs)
    else:
      # Otherwise, use the file name without extension
      final_path_segment = path_after_docs.removesuffix(".md")

    if final_path_segment and not final_path_segment.endswith("/"):
      final_path_segment += "/"

    potential_url = urljoin(base_url, final_path_segment)

    # Check if the generated link exists
    if _check_url_exists(potential_url):
      return potential_url
    else:
      # If it doesn't exist, fall back to the regular github url
      return _generate_github_url(prefix, relative_path)

  # Convert the links for other cases, e.g. adk-python
  else:
    return _generate_github_url(prefix, relative_path)


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
    if event.content and event.content.parts:
      if text := "".join(part.text or "" for part in event.content.parts):
        if event.author != "user":
          final_response_text += text

  return final_response_text
