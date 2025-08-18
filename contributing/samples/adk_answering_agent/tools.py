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

from typing import Any
from typing import Dict
from typing import Optional

from adk_answering_agent.settings import OWNER
from adk_answering_agent.settings import REPO
from adk_answering_agent.utils import convert_gcs_to_https
from adk_answering_agent.utils import error_response
from adk_answering_agent.utils import run_graphql_query
import requests


def get_discussion_and_comments(discussion_number: int) -> dict[str, Any]:
  """Fetches a discussion and its comments using the GitHub GraphQL API.

  Args:
      discussion_number: The number of the GitHub discussion.

  Returns:
      A dictionary with the request status and the discussion details.
  """
  print(f"Attempting to get discussion #{discussion_number} and its comments")
  query = """
        query($owner: String!, $repo: String!, $discussionNumber: Int!) {
          repository(owner: $owner, name: $repo) {
            discussion(number: $discussionNumber) {
              id
              title
              body
              createdAt
              closed
              author {
                login
              }
              # For each discussion, fetch the latest 20 labels.
              labels(last: 20) {
                nodes {
                  id
                  name
                }
              }
              # For each discussion, fetch the latest 100 comments.
              comments(last: 100) {
                nodes {
                  id
                  body
                  createdAt
                  author {
                    login
                  }
                  # For each discussion, fetch the latest 50 replies
                  replies(last: 50) {
                    nodes {
                      id
                      body
                      createdAt
                      author {
                        login
                      }
                    }
                  }
                }
              }
            }
          }
        }
    """
  variables = {
      "owner": OWNER,
      "repo": REPO,
      "discussionNumber": discussion_number,
  }
  try:
    response = run_graphql_query(query, variables)
    if "errors" in response:
      return error_response(str(response["errors"]))
    discussion_data = (
        response.get("data", {}).get("repository", {}).get("discussion")
    )
    if not discussion_data:
      return error_response(f"Discussion #{discussion_number} not found.")
    return {"status": "success", "discussion": discussion_data}
  except requests.exceptions.RequestException as e:
    return error_response(str(e))


def add_comment_to_discussion(
    discussion_id: str, comment_body: str
) -> dict[str, Any]:
  """Adds a comment to a specific discussion.

  Args:
      discussion_id: The GraphQL node ID of the discussion.
      comment_body: The content of the comment in Markdown.

  Returns:
      The status of the request and the new comment's details.
  """
  print(f"Adding comment to discussion {discussion_id}")
  query = """
        mutation($discussionId: ID!, $body: String!) {
          addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
            comment {
              id
              body
              createdAt
              author {
                login
              }
            }
          }
        }
    """
  if not comment_body.startswith("**Response from ADK Answering Agent"):
    comment_body = (
        "**Response from ADK Answering Agent (experimental, answer may be"
        " inaccurate)**\n\n"
        + comment_body
    )

  variables = {"discussionId": discussion_id, "body": comment_body}
  try:
    response = run_graphql_query(query, variables)
    if "errors" in response:
      return error_response(str(response["errors"]))
    new_comment = (
        response.get("data", {}).get("addDiscussionComment", {}).get("comment")
    )
    return {"status": "success", "comment": new_comment}
  except requests.exceptions.RequestException as e:
    return error_response(str(e))


def get_label_id(label_name: str) -> str | None:
  """Helper function to find the GraphQL node ID for a given label name."""
  print(f"Finding ID for label '{label_name}'...")
  query = """
    query($owner: String!, $repo: String!, $labelName: String!) {
      repository(owner: $owner, name: $repo) {
        label(name: $labelName) {
          id
        }
      }
    }
    """
  variables = {"owner": OWNER, "repo": REPO, "labelName": label_name}

  try:
    response = run_graphql_query(query, variables)
    if "errors" in response:
      print(
          f"[Warning] Error from GitHub API response for label '{label_name}':"
          f" {response['errors']}"
      )
      return None
    label_info = response["data"].get("repository", {}).get("label")
    if label_info:
      return label_info.get("id")
    print(f"[Warning] Label information for '{label_name}' not found.")
    return None
  except requests.exceptions.RequestException as e:
    print(f"[Warning] Error from GitHub API: {e}")
    return None


def add_label_to_discussion(
    discussion_id: str, label_name: str
) -> dict[str, Any]:
  """Adds a label to a specific discussion.

  Args:
      discussion_id: The GraphQL node ID of the discussion.
      label_name: The name of the label to add (e.g., "bug").

  Returns:
      The status of the request and the label details.
  """
  print(
      f"Attempting to add label '{label_name}' to discussion {discussion_id}..."
  )
  # First, get the GraphQL ID of the label by its name
  label_id = get_label_id(label_name)
  if not label_id:
    return error_response(f"Label '{label_name}' not found.")

  # Then, perform the mutation to add the label to the discussion
  mutation = """
    mutation AddLabel($discussionId: ID!, $labelId: ID!) {
      addLabelsToLabelable(input: {labelableId: $discussionId, labelIds: [$labelId]}) {
        clientMutationId
      }
    }
    """
  variables = {"discussionId": discussion_id, "labelId": label_id}
  try:
    response = run_graphql_query(mutation, variables)
    if "errors" in response:
      return error_response(str(response["errors"]))
    return {"status": "success", "label_id": label_id, "label_name": label_name}
  except requests.exceptions.RequestException as e:
    return error_response(str(e))


def convert_gcs_links_to_https(gcs_uris: list[str]) -> Dict[str, Optional[str]]:
  """Converts GCS files link into publicly accessible HTTPS links.

  Args:
      gcs_uris: A list of GCS files links, in the format
        'gs://bucket_name/prefix/relative_path'.

  Returns:
      A dictionary mapping the original GCS files links to the converted HTTPS
      links. If a GCS link is invalid, the corresponding value in the dictionary
      will be None.
  """
  return {gcs_uri: convert_gcs_to_https(gcs_uri) for gcs_uri in gcs_uris}
