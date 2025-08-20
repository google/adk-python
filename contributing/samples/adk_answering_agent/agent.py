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

from adk_answering_agent.gemini_assistant.agent import root_agent as gemini_assistant_agent
from adk_answering_agent.settings import BOT_RESPONSE_LABEL
from adk_answering_agent.settings import IS_INTERACTIVE
from adk_answering_agent.settings import OWNER
from adk_answering_agent.settings import REPO
from adk_answering_agent.settings import VERTEXAI_DATASTORE_ID
from adk_answering_agent.tools import add_comment_to_discussion
from adk_answering_agent.tools import add_label_to_discussion
from adk_answering_agent.tools import convert_gcs_links_to_https
from adk_answering_agent.tools import get_discussion_and_comments
from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = (
      "Ask for user approval or confirmation for adding the comment."
  )
else:
  APPROVAL_INSTRUCTION = (
      "**Do not** wait or ask for user approval or confirmation for adding the"
      " comment."
  )

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
      * If you need infromation about Gemini API, ask the `gemini_assistant` agent to provide the information and references.
      * You can call the `gemini_assistant` agent with multiple queries to find all the relevant information.
    5. If you can find relevant information, use the `add_comment_to_discussion` tool to add a comment to the discussion.
    6. If you post a comment, add the label {BOT_RESPONSE_LABEL} to the discussion using the `add_label_to_discussion` tool.

    IMPORTANT:
      * {APPROVAL_INSTRUCTION}
      * Your response should be based on the information you found in the document store. Do not invent
        information that is not in the document store. Do not invent citations which are not in the document store.
      * **Be Objective**: your answer should be based on the facts you found in the document store, do not be misled by user's assumptions or user's understanding of ADK.
      * If you can't find the answer or information in the document store, **do not** respond.
      * Start with a short summary of your response in the comment as a TLDR, e.g. "**TLDR**: <your summary>".
      * Have a divider line between the TLDR and your detail response.
      * Do not respond to any other discussion except the one specified by the user.
      * Please include your justification for your decision in your output
        to the user who is telling with you.
      * If you uses citation from the document store, please provide a footnote
        referencing the source document format it as: "[1] publicly accessible HTTPS URL of the document".
        * You **should always** use the `convert_gcs_links_to_https` tool to convert GCS links (e.g. "gs://...") to HTTPS links.
        * **Do not** use the `convert_gcs_links_to_https` tool for non-GCS links.
        * Make sure the citation URL is valid. Otherwise do not list this specific citation.
    """,
    tools=[
        VertexAiSearchTool(data_store_id=VERTEXAI_DATASTORE_ID),
        AgentTool(gemini_assistant_agent),
        get_discussion_and_comments,
        add_comment_to_discussion,
        add_label_to_discussion,
        convert_gcs_links_to_https,
    ],
)
