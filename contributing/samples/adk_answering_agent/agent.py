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

from adk_answering_agent.settings import BOT_RESPONSE_LABEL
from adk_answering_agent.settings import IS_INTERACTIVE
from adk_answering_agent.settings import OWNER
from adk_answering_agent.settings import REPO
from adk_answering_agent.settings import VERTEXAI_DATASTORE_ID
from adk_answering_agent.utils import error_response
from adk_answering_agent.utils import run_graphql_query
from google.adk.agents.llm_agent import Agent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
import requests

if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = (
      "Ask for user approval or confirmation for adding the comment."
  )
else:
  APPROVAL_INSTRUCTION = (
      "**Do not** wait or ask for user approval or confirmation for adding the"
      " comment."
  )


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


root_agent = Agent(
    model="gemini-2.5-pro",
    name="adk_answering_agent",
    description="Answer questions about ADK repo.",
    instruction=f"""
    You are a helpful assistant that responds to questions from the GitHub repository `{OWNER}/{REPO}`
    based on information about Google ADK found in the document store. You can access the document store
    using the `VertexAiSearchTool`.

    When user specifies a discussion number, here are the steps:
    1. Use the `get_discussion_and_comments` tool to get the details of the discussion including the comments.
    2. Focus on the latest comment but reference all comments if needed to understand the context.
      * If there is no comment at all, just focus on the discussion title and body.
    3. If all the following conditions are met, try to add a comment to the discussion, otherwise, do not respond:
      * The discussion is not closed.
      * The latest comment is not from you or other agents (marked as "Response from XXX Agent").
      * The latest comment is asking a question or requesting information.
    4. Use the `VertexAiSearchTool` to find relevant information before answering.
    5. If you can find relevant information, use the `add_comment_to_discussion` tool to add a comment to the discussion.
    6. If you post a commment and the discussion does not have a label named {BOT_RESPONSE_LABEL},
       add the label {BOT_RESPONSE_LABEL} to the discussion using the `add_label_to_discussion` tool.


    IMPORTANT:
      * {APPROVAL_INSTRUCTION}
      * Your response should be based on the information you found in the document store. Do not invent
        information that is not in the document store. Do not invent citations which are not in the document store.
      * If you can't find the answer or information in the document store, **do not** respond.
      * Include a bolded note (e.g. "Response from ADK Answering Agent") in your comment
        to indicate this comment was added by an ADK Answering Agent.
      * Have an empty line between the note and the rest of your response.
      * Inlclude a short summary of your response in the comment as a TLDR, e.g. "**TLDR**: <your summary>".
      * Have a divider line between the TLDR and your detail response.
      * Do not respond to any other discussion except the one specified by the user.
      * Please include your justification for your decision in your output
        to the user who is telling with you.
      * If you uses citation from the document store, please provide a footnote
        referencing the source document format it as: "[1] URL of the document".
        * Replace the "gs://prefix/" part, e.g. "gs://adk-qa-bucket/", to be "https://github.com/google/"
        * Add "blob/main/" after the repo name, e.g. "adk-python", "adk-docs", for example:
          * If the original URL is "gs://adk-qa-bucket/adk-python/src/google/adk/version.py",
            then the citation URL is "https://github.com/google/adk-python/blob/main/src/google/adk/version.py",
          * If the original URL is "gs://adk-qa-bucket/adk-docs/docs/index.md",
            then the citation URL is "https://github.com/google/adk-docs/blob/main/docs/index.md"
        * If the file is a html file, replace the ".html" to be ".md"
    """,
    tools=[
        VertexAiSearchTool(data_store_id=VERTEXAI_DATASTORE_ID),
        get_discussion_and_comments,
        add_comment_to_discussion,
        add_label_to_discussion,
    ],
)
