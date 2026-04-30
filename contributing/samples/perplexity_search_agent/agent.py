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

"""Sample agent using the Perplexity Search tool.

Set the PERPLEXITY_API_KEY environment variable before running this agent.
See https://docs.perplexity.ai/api-reference/search-post for API details.
"""

from google.adk import Agent
from google.adk.tools.perplexity_search_tool import PerplexitySearchTool

perplexity_search = PerplexitySearchTool(max_results=5)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description=(
        'an agent whose job it is to answer questions by searching the web'
        ' via the Perplexity Search API.'
    ),
    instruction=(
        'You are an agent whose job is to answer questions by searching the'
        ' web with the perplexity_search tool. Cite the URLs of the sources'
        ' you used in your final answer.'
    ),
    tools=[perplexity_search],
)
