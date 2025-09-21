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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools.youtube.search import YouTubeSearchTool

# Instantiate the tool
langchain_yt_tool = YouTubeSearchTool()

# Wrap the tool in the LangchainTool class from ADK
adk_yt_tool = LangchainTool(
    tool=langchain_yt_tool,
)

root_agent = LlmAgent(
    name="youtube_search_agent",
    model="gemini-2.0-flash",  # Replace with the actual model name
    instruction="""
    Ask customer to provide singer name, and the number of videos to search.
    """,
    description="Help customer to search for a video on Youtube.",
    tools=[adk_yt_tool],
    output_key="youtube_search_output",
)
