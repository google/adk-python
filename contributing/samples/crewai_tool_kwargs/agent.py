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

"""Sample demonstrating CrewAI tool with **kwargs parameter handling.

This sample shows how CrewaiTool correctly passes arbitrary parameters
through **kwargs, which is a common pattern in CrewAI tools.
"""

from typing import Optional

from crewai.tools import BaseTool
from google.adk import Agent
from google.adk.tools.crewai_tool import CrewaiTool
from pydantic import BaseModel
from pydantic import Field


class SearchInput(BaseModel):
  """Input schema for the search tool."""

  query: str = Field(..., description="The search query string")
  category: Optional[str] = Field(
      None, description="Filter by category (e.g., 'technology', 'science')"
  )
  date_range: Optional[str] = Field(
      None, description="Filter by date range (e.g., 'last_week', '2024')"
  )
  limit: Optional[int] = Field(
      None, description="Limit the number of results (e.g., 10, 20)"
  )


class CustomSearchTool(BaseTool):
  """A custom CrewAI tool that accepts arbitrary search parameters via **kwargs.

  This demonstrates the key CrewAI tool pattern where tools accept
  flexible parameters through **kwargs.
  """

  name: str = "custom_search"
  description: str = (
      "Search for information with flexible filtering options. "
      "Accepts a query and optional filter parameters like category, "
      "date_range, limit, etc."
  )
  args_schema: type[BaseModel] = SearchInput

  def _run(self, query: str, **kwargs) -> str:
    """Execute search with arbitrary filter parameters.

    Args:
      query: The search query string.
      **kwargs: Additional filter parameters like category, date_range, limit.

    Returns:
      A formatted string showing the query and applied filters.
    """
    result_parts = [f"Searching for: '{query}'"]

    if kwargs:
      result_parts.append("Applied filters:")
      for key, value in kwargs.items():
        result_parts.append(f"  - {key}: {value}")
    else:
      result_parts.append("No additional filters applied.")

    # Simulate search results
    result_parts.append(f"\nFound 3 results matching your criteria.")

    return "\n".join(result_parts)


crewai_search_tool = CustomSearchTool()

# Wrap it with ADK's CrewaiTool
adk_search_tool = CrewaiTool(
    crewai_search_tool,
    name="search_with_filters",
    description=(
        "Search for information with optional filters like category, "
        "date_range, or limit"
    ),
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="search_agent",
    description="An agent that can search with flexible filtering options",
    instruction="""
      You are a helpful search assistant.
      When users ask you to search, use the search_with_filters tool.
      You can pass additional parameters like:
      - category: to filter by category (e.g., "technology", "science")
      - date_range: to filter by date (e.g., "last_week", "2024")
      - limit: to limit the number of results (e.g., 10, 20)

      Always acknowledge what filters you're applying.
    """,
    tools=[adk_search_tool],
)
