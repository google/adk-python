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

from google.adk import Agent
from google.adk.tools.google_search_tool import google_search

root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='root_agent',
    description="""an agent whose job it is to perform Google search queries and answer questions about the results.""",
    instruction="""You are an agent whose job is to perform Google search queries and answer questions about the results.
""",
    tools=[google_search],
)
