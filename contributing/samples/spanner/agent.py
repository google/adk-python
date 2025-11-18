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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.spanner.settings import Capabilities
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
import google.adk.tools.spanner.utils as spanner_tool_utils
from google.adk.tools.tool_context import ToolContext
import google.auth
from google.auth.credentials import Credentials
from google.cloud.spanner_v1 import param_types as spanner_param_types

# Define an appropriate credential type
# Set to None to use the application default credentials (ADC) for a quick
# development.
CREDENTIALS_TYPE = None


# Define Spanner tool config with read capability set to allowed.
tool_settings = SpannerToolSettings(capabilities=[Capabilities.DATA_READ])

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = SpannerCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
      scopes=[
          "https://www.googleapis.com/auth/spanner.admin",
          "https://www.googleapis.com/auth/spanner.data",
      ],
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = SpannerCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = SpannerCredentialsConfig(
      credentials=application_default_credentials
  )

# Example 1: Use tools from the Spanner toolset.
# For example, data exploration agents help the Spanner database developer or
# data engineer of the organization.
spanner_toolset = SpannerToolset(
    credentials_config=credentials_config,
    spanner_tool_settings=tool_settings,
    # Uncomment to explicitly specify allowed tools.
    # tool_filter=["execute_sql", "get_table_schema"],
)


# Replace the following settings with your specific Spanner database for example
# 2 and 3.
# For example, these settings can also be read from a configuration file or
# environment variables.
_SPANNER_PROJECT_ID = "<PROJECT_ID>"
_SPANNER_INSTANCE_ID = "<INSTANCE_ID>"
_SPANNER_DATABASE_ID = "<DATABASE_ID>"


# Example 2: Create a customized Spanner query tool with a template SQL query.
# Note that this approach makes it **more vulnerable to SQL injection**. This
# might be suitable for some specific use cases, and **adding additional checks
# or callbacks** is recommended.
def count_rows_in_table(
    table_name: str,
    credentials: Credentials,
    settings: SpannerToolSettings,
    tool_context: ToolContext,
):
  """Counts the total number of rows for a specified table.

  Args:
    table_name: The name of the table for which to count rows.

  Returns:
      The total number of rows in the table.
  """

  # Example of adding additional checks:
  # if table_name not in ["table1", "table2"]:
  #   raise ValueError("Table name is not allowed.")

  sql_template = f"SELECT COUNT(*) FROM {table_name}"

  return spanner_tool_utils.execute_sql(
      project_id=_SPANNER_PROJECT_ID,
      instance_id=_SPANNER_INSTANCE_ID,
      database_id=_SPANNER_DATABASE_ID,
      query=sql_template,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
  )


# Example 3: Create a customized Spanner query tool with a template
# parameterized SQL query.
# For example, it could query data that all authenticated users of the system
# have access to. This can also work for searching public knowledge bases, such
# as company policies and FAQs.
def search_hotels(
    location_name: str,
    credentials: Credentials,
    settings: SpannerToolSettings,
    tool_context: ToolContext,
):
  """Search hotels for a specific location.

  This function takes a geographical location name and returns a list of hotels
  in that area, including key details for each.

  Args:
    location_name (str): The geographical location (e.g., city or town) for the
                         hotel search.
    Example:
    {
        "location_name": "Seattle"
    }
    Example:
    {
        "location_name": "New York"
    }
    Example:
    {
        "location_name": "Los Angeles"
    }

  Returns:
      The hotels name, rating and description.
  """

  sql_template = """
      SELECT name, rating, description FROM hotels
      WHERE location_name = @location_name
      """
  return spanner_tool_utils.execute_sql(
      project_id=_SPANNER_PROJECT_ID,
      instance_id=_SPANNER_INSTANCE_ID,
      database_id=_SPANNER_DATABASE_ID,
      query=sql_template,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
      params={"location_name": location_name},
      params_types={"location_name": spanner_param_types.STRING},
  )


# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="spanner_agent",
    description=(
        "Agent to answer questions about Spanner database tables and"
        " execute SQL queries."
    ),
    instruction="""\
        You are a data agent with access to several Spanner tools.
        Make use of those tools to answer the user's questions.
    """,
    tools=[
        # Use tools from Spanner toolset.
        spanner_toolset,
        # Or, uncomment to use customized Spanner tools.
        # GoogleTool(
        #     func=count_rows_in_table,
        #     credentials_config=credentials_config,
        #     tool_settings=tool_settings,
        # ),
        # GoogleTool(
        #     func=search_hotels,
        #     credentials_config=credentials_config,
        #     tool_settings=tool_settings,
        # ),
    ],
)
