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


import logging

from .google_api_tool_set import GoogleApiToolSet

logger = logging.getLogger(__name__)

calendar_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="calendar",
    api_version="v3",
)

bigquery_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="bigquery",
    api_version="v2",
)

gmail_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="gmail",
    api_version="v1",
)

youtube_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="youtube",
    api_version="v3",
)

slides_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="slides",
    api_version="v1",
)

sheets_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="sheets",
    api_version="v4",
)

docs_tool_set = GoogleApiToolSet.load_tool_set(
    api_name="docs",
    api_version="v1",
)

# Comprehensive usage instructions for BigQuery

def setup_bigquery_connector():
    """
    Detailed steps for setting up a BigQuery connector and granting IAM roles.

    1. Go to the IAM & Admin page in the Google Cloud Console.
    2. Select your project.
    3. Click on the "Add" button to add a new member.
    4. Enter the service account email address.
    5. Assign the following roles:
       - BigQuery Data Viewer
       - BigQuery User
       - BigQuery Job User
    6. Click "Save" to apply the changes.
    7. Go to the Integration Connectors page in the Google Cloud Console.
    8. Select your project and location.
    9. Click on the "Create Connection" button.
    10. Follow the prompts to configure your connection, including specifying the connection name, service name, and host.
    11. Ensure that the connection is in the same region as your Application Integration.
    12. Create an integration named "ExecuteConnection" with a trigger "api_trigger/ExecuteConnection".
    13. Use the connection name, service name, and host in your integration configuration.
    """

def example_bigquery_operations():
    """
    Example code snippets for performing BigQuery operations using the ADK.

    from google.adk.agents import Agent
    from google.adk.tools.google_api_tool.google_api_tool_sets import bigquery_tool_set

    # Create an agent with BigQuery tools
    bigquery_agent = Agent(
        name="bigquery_agent",
        model="gemini-2.0-flash-exp",
        instruction="You are a BigQuery assistant. Help the user with BigQuery operations.",
        description="An assistant for BigQuery operations.",
        tools=bigquery_tool_set.get_tools()
    )

    # Example usage of BigQuery tool
    async def query_bigquery():
        query = "SELECT * FROM `project.dataset.table` LIMIT 10"
        result = await bigquery_agent.run_async(
            args={"query": query},
            tool_context=None
        )
        print(result)

    # Run the example
    import asyncio
    asyncio.run(query_bigquery())
    """
