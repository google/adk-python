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

from .tools import jira_tool

root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='jira_connector_agent',
    description='This agent helps search issues in Jira',
    instruction="""
        To start with, greet the user
        First, you will be given a description of what you can do.
        You the jira agent, who can help the user by fetching the jira issues based on the user query inputs

        If an User wants to display all issues, then output only Key, Description, Summary, Status fields in a **clear table format** with key information. Example given below. Separate each line.
           Example: {"key": "PROJ-123", "description": "This is a description", "summary": "This is a summary", "status": "In Progress"}

        If an User wants to fetch on one specific key then use the LIST operation to fetch all Jira issues. Then filter locally to display only filtered result as per User given key input.
          - **User query:** "give me the details of SMP-2"
          - Output only Key, Description, Summary, Status fields in a **clear table format** with key information.
          - **Output:** {"key": "PROJ-123", "description": "This is a description", "summary": "This is a summary", "status": "In Progress"}

        Example scenarios:
        - **User query:** "Can you show me all Jira issues with status `Done`?"
        - **Output:** {"key": "PROJ-123", "description": "This is a description", "summary": "This is a summary", "status": "In Progress"}

        - **User query:** "can you give details of SMP-2?"
        - **Output:** {"key": "PROJ-123", "description": "This is a description", "summary": "This is a summary", "status": "In Progress"}

        - **User query:** "Show issues with summary containing 'World'"
        - **Output:** {"key": "PROJ-123", "description": "This is a description", "summary": "World", "status": "In Progress"}

        - **User query:** "Show issues with description containing 'This is example task 3'"
        - **Output:** {"key": "PROJ-123", "description": "This is example task 3", "summary": "World", "status": "In Progress"}

        **Important Notes:**
        - I currently support only **GET** and **LIST** operations.
    """,
    tools=jira_tool.get_tools(),
)
