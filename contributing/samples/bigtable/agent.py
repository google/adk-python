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

import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.bigtable import query_tool as bigtable_query_tool
from google.adk.tools.bigtable.bigtable_credentials import BigtableCredentialsConfig
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset
from google.adk.tools.bigtable.settings import BigtableToolSettings
from google.adk.tools.google_tool import GoogleTool
import google.auth
from google.cloud.bigtable.data.execute_query.metadata import SqlType

# Define an appropriate credential type.
# None for Application Default Credentials
CREDENTIALS_TYPE = None

# Define Bigtable tool config with read capability set to allowed.
tool_settings = BigtableToolSettings()

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = BigtableCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
      scopes=[
          "https://www.googleapis.com/auth/bigtable.admin",
          "https://www.googleapis.com/auth/bigtable.data",
      ],
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = BigtableCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = BigtableCredentialsConfig(
      credentials=application_default_credentials
  )

bigtable_toolset = BigtableToolset(
    credentials_config=credentials_config, bigtable_tool_settings=tool_settings
)

_BIGTABLE_PROJECT_ID = "google.com:cloud-bigtable-dev"
_BIGTABLE_INSTANCE_ID = "annenguyen-bus-instance"


def search_hotels_by_location(
    location_name: str,
    credentials: google.auth.credentials.Credentials,
    settings: BigtableToolSettings,
    tool_context: google.adk.tools.tool_context.ToolContext,
):
  """Search hotels by location name.

  This function takes a location name and returns a list of hotels
  in that area.

  Args:
    location_name (str): The geographical location (e.g., city or town) for the
      hotel search.
    Example: { "location_name": "Basel" }

  Returns:
      The hotels name, price tier.
  """

  sql_template = """
  SELECT
    TO_INT64(cf['id']) as id,
    CAST(cf['name'] AS STRING) AS name,
    CAST(cf['location'] AS STRING) AS location,
    CAST(cf['price_tier'] AS STRING) AS price_tier,
    CAST(cf['checkin_date'] AS STRING) AS checkin_date,
    CAST(cf['checkout_date'] AS STRING) AS checkout_date
  FROM hotels
  WHERE LOWER(CAST(cf['location'] AS STRING)) LIKE LOWER(CONCAT('%', @location_name, '%'))
  """
  return bigtable_query_tool.execute_sql(
      project_id=_BIGTABLE_PROJECT_ID,
      instance_id=_BIGTABLE_INSTANCE_ID,
      query=sql_template,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
      parameters={"location": location_name},
      parameter_types={"location": SqlType.String()},
  )


# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="bigtable_agent",
    description=(
        "Agent to answer questions about Bigtable database tables and"
        " execute SQL queries."
    ),  # TODO(b/360128447): Update description
    instruction="""\
        You are a data agent with access to several Bigtable tools.
        Make use of those tools to answer the user's questions.
    """,
    tools=[
        bigtable_toolset,
        # Or, uncomment to use customized Bigtable tools.
        # GoogleTool(
        #     func=search_hotels_by_location,
        #     credentials_config=credentials_config,
        #     tool_settings=tool_settings,
        # ),
    ],
)
