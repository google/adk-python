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
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import dict_to_auth_scheme
from google.genai import types

# Load environment variables from .env file
load_dotenv()

connection_name = os.getenv("CONNECTION_NAME")
connection_project = os.getenv("CONNECTION_PROJECT")
connection_location = os.getenv("CONNECTION_LOCATION")
client_secret = os.getenv("CLIENT_SECRET")
client_id = os.getenv("CLIENT_ID")


oauth2_data_google_cloud = {
    "type": "oauth2",
    "flows": {
        "authorizationCode": {
            "authorizationUrl": "https://accounts.google.com/o/oauth2/auth",
            "tokenUrl": "https://oauth2.googleapis.com/token",
            "scopes": {
                "https://www.googleapis.com/auth/cloud-platform": (
                    "View and manage your data across Google Cloud Platform"
                    " services"
                ),
                "https://www.googleapis.com/auth/calendar.readonly": (
                    "View your calendars"
                ),
            },
        }
    },
}

oauth2_scheme = dict_to_auth_scheme(oauth2_data_google_cloud)

auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id=client_id,
        client_secret=client_secret,
    ),
)

calendar_tool = ApplicationIntegrationToolset(
    project=connection_project,
    location=connection_location,
    tool_name_prefix="calendar_tool",
    connection=connection_name,
    actions=["GET_calendars/%7BcalendarId%7D/events"],
    tool_instructions="""
  Use this tool to list events in a calendar. Get calendarId from the user and use it in tool as following example:
  connectorInputPayload: { "Path parameters": { "calendarId": "primary" } }. Follow the schema correctly. Note its "Path parameters" and not "Path_parameters".
    """,
    auth_scheme=oauth2_scheme,
    auth_credential=auth_credential,
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="data_processing_agent",
    description="Agent that can list events in a calendar.",
    instruction="""
      Helps you with calendar related tasks.
    """,
    tools=calendar_tool.get_tools(),
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)
