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


from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools.youtube.search import YouTubeSearchTool

# Instantiate the tool
langchain_yt_tool = YouTubeSearchTool()

# Wrap the tool in the LangchainTool class from ADK
adk_yt_tool = LangchainTool(
    tool=langchain_yt_tool,
)

youtube_search_agent = Agent(
    name="youtube_search_agent",
    model="gemini-2.0-flash",  # Replace with the actual model name
    instruction="""
    Ask customer to provide singer name, and the number of videos to search.
    """,
    description="Help customer to search for a video on Youtube.",
    tools=[adk_yt_tool],
    output_key="youtube_search_output",
)

bigquery_agent = RemoteA2aAgent(
    name="bigquery_agent",
    description="Help customer to manage notion workspace.",
    agent_card=(
        f"http://localhost:8001/a2a/bigquery_agent{AGENT_CARD_WELL_KNOWN_PATH}"
    ),
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="root_agent",
    instruction="""
      You are a helpful assistant that can help search youtube videos, look up BigQuery datasets and tables.
      You delegate youtube search tasks to the youtube_search_agent.
      You delegate BigQuery tasks to the bigquery_agent.
      Always clarify the results before proceeding.
    """,
    global_instruction=(
        "You are a helpful assistant that can help search youtube videos, look"
        " up BigQuery datasets and tables."
    ),
    sub_agents=[youtube_search_agent, bigquery_agent],
)
