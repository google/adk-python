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

"""Sample agent using Application Integration toolset."""

import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset

# Load environment variables from .env file
load_dotenv()

connection_name = os.getenv("CONNECTION_NAME")
connection_project = os.getenv("CONNECTION_PROJECT")
connection_location = os.getenv("CONNECTION_LOCATION")


jira_toolset = ApplicationIntegrationToolset(
    project=connection_project,
    location=connection_location,
    connection=connection_name,
    entity_operations={"Issues": [], "Projects": []},
    tool_name_prefix="jira_issue_manager",
)

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="Issue_Management_Agent",
    instruction="""
    You are an agent that helps manage issues in a JIRA instance.
    Be accurate in your responses based on the tool response. You can perform any formatting in the response that is appropriate or if asked by the user.
    If there is an error in the tool response, understand the error and try and see if you can fix the error and then  and execute the tool again. For example if a variable or parameter is missing, try and see if you can find it in the request or user query or default it and then execute the tool again or check for other tools that could give you the details.
    If there are any math operations like count or max, min in the user request, call the tool to get the data and perform the math operations and then return the result in the response. For example for maximum, fetch the list and then do the math operation.
    """,
    tools=[jira_toolset],
)
