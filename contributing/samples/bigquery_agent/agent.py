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

import os

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.tools.google_api_tool import BigQueryToolset

# Load environment variables from .env file
load_dotenv()

# Access the variable
oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")
tools_to_expose = [
    "bigquery_datasets_list",
    "bigquery_datasets_get",
    "bigquery_datasets_insert",
    "bigquery_tables_list",
    "bigquery_tables_get",
    "bigquery_tables_insert",
]
bigquery_toolset = BigQueryToolset(
    client_id=oauth_client_id,
    client_secret=oauth_client_secret,
    tool_filter=tools_to_expose,
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="bigquery_agent",
    instruction="""
      You are a helpful Google BigQuery agent that help to manage users' data on Google BigQuery.
      Use the provided tools to conduct various operations on users' data in Google BigQuery.

      Scenario 1:
      The user wants to query their biguqery datasets
      Use bigquery_datasets_list to query user's datasets

      Scenario 2:
      The user wants to query the details of a specific dataset
      Use bigquery_datasets_get to get a dataset's details

      Scenario 3:
      The user wants to create a new dataset
      Use bigquery_datasets_insert to create a new dataset

      Scenario 4:
      The user wants to query their tables in a specific dataset
      Use bigquery_tables_list to list all tables in a dataset

      Scenario 5:
      The user wants to query the details of a specific table
      Use bigquery_tables_get to get a table's details

      Scenario 6:
      The user wants to insert a new table into a dataset
      Use bigquery_tables_insert to insert a new table into a dataset

      Current user:
      <User>
      {userInfo?}
      </User>
""",
    tools=[bigquery_toolset],
)
