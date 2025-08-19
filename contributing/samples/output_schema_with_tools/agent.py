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

"""Sample agent demonstrating output_schema with tools feature.

This agent shows how to use structured output (output_schema) alongside
other tools. Previously, this combination was not allowed, but now it's
supported through a workaround that uses a special set_model_response tool.
"""

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field
import requests


class PersonInfo(BaseModel):
  """Structured information about a person."""

  name: str = Field(description="The person's full name")
  age: int = Field(description="The person's age in years")
  occupation: str = Field(description="The person's job or profession")
  location: str = Field(description="The city and country where they live")
  biography: str = Field(description="A brief biography of the person")


def search_wikipedia(query: str) -> str:
  """Search Wikipedia for information about a topic.

  Args:
    query: The search query to look up on Wikipedia

  Returns:
    Summary of the Wikipedia article if found, or error message if not found
  """
  try:
    # Use Wikipedia API to search for the article
    search_url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + query.replace(" ", "_")
    )
    response = requests.get(search_url, timeout=10)

    if response.status_code == 200:
      data = response.json()
      return (
          f"Title: {data.get('title', 'N/A')}\n\nSummary:"
          f" {data.get('extract', 'No summary available')}"
      )
    else:
      return (
          f"Wikipedia article not found for '{query}'. Status code:"
          f" {response.status_code}"
      )

  except Exception as e:
    return f"Error searching Wikipedia: {str(e)}"


def get_current_year() -> str:
  """Get the current year.

  Returns:
    The current year as a string
  """
  from datetime import datetime

  return str(datetime.now().year)


# Create the agent with both output_schema and tools
root_agent = LlmAgent(
    name="person_info_agent",
    model="gemini-2.5-pro",
    instruction="""
You are a helpful assistant that gathers information about famous people.

When asked about a person, you should:
1. Use the search_wikipedia tool to find information about them
2. Use the get_current_year tool if you need to calculate ages
3. Compile the information into a structured response using the PersonInfo format

Always use the set_model_response tool to provide your final answer in the required structured format.
    """.strip(),
    output_schema=PersonInfo,
    tools=[
        search_wikipedia,
        get_current_year,
    ],
)
