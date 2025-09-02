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

"""Sub-agent for Google Search functionality."""

from google.adk.agents import LlmAgent
from google.adk.tools import google_search


def create_google_search_agent() -> LlmAgent:
  """Create a sub-agent that only uses google_search tool."""
  return LlmAgent(
      name="google_search_agent",
      description=(
          "Agent for performing Google searches to find ADK examples and"
          " documentation"
      ),
      instruction="""You are a specialized search agent for the Agent Builder Assistant.

Your role is to search for relevant ADK (Agent Development Kit) examples, patterns, documentation, and solutions.

When given a search query, use the google_search tool to find:
- ADK configuration examples and patterns
- Multi-agent system architectures and workflows
- Best practices and documentation
- Similar use cases and implementations
- Troubleshooting solutions and error fixes
- API references and implementation guides

SEARCH STRATEGIES:
- Use site-specific searches for targeted results:
  * "site:github.com/google/adk-python [query]" for core ADK examples
  * "site:github.com/google/adk-samples [query]" for sample implementations
  * "site:github.com/google/adk-docs [query]" for documentation
- Use general searches for broader community solutions
- Search for specific agent types, tools, or error messages
- Look for configuration patterns and architectural approaches

Return the search results with:
1. Relevant URLs found
2. Brief description of what each result contains
3. Relevance to the original query
4. Suggestions for which URLs should be fetched for detailed analysis

Focus on finding practical, actionable examples that can guide ADK development and troubleshooting.""",
      model="gemini-2.5-flash",
      tools=[google_search],
  )
