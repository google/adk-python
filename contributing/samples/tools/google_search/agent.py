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

"""Native Google Search Grounding Agent with Gemini in ADK 2.0."""

from google.adk import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="google_search_agent",
    description=(
        "A research assistant that answers current-events and factual "
        "questions using native Google Search grounding."
    ),
    instruction="""
      You are a helpful research assistant.
      Use Google Search when the user asks questions that depend on recent,
      time-sensitive, or specialized public information.
      Synthesize a clear and concise response and cite the sources used.
    """,
    tools=[google_search],
)
