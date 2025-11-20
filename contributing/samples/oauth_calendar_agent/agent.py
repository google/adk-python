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

from datetime import datetime
import os

from dotenv import load_dotenv
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.tools.google_api_tool import CalendarToolset
from google.adk.tools.tool_context import ToolContext
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Load environment variables from .env file
load_dotenv()

# Access the variable
oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")


SCOPES = ["https://www.googleapis.com/auth/calendar"]

calendar_toolset = CalendarToolset(
    # you can also replace below customized `list_calendar_events` with build-in
    # google calendar tool by adding `calendar_events_list` in the filter list
    client_id=oauth_client_id,
    client_secret=oauth_client_secret,
    tool_filter=["calendar_events_get", "calendar_events_update"],
    tool_name_prefix="google",
)


# this tool will be invoked right after google_calendar_events_get returns a
# final response to test whether adk works correctly for subsequent function
# call right after a function call that request auth
# see https://github.com/google/adk-python/issues/1944 for details
def redact_event_content(event_content: str) -> str:
  """Redact confidential information in the calendar event content
  Args:
      event_content: the content of the calendar event to redact

  Returns:
      str: redacted content of the calendar event
  """
  return event_content


def list_calendar_events(
    start_time: str,
    end_time: str,
    limit: int,
    tool_context: ToolContext,
    credential: AuthCredential,
) -> list[dict]:
  """Search for calendar events.

  Example:

      flights = get_calendar_events(
          calendar_id='joedoe@gmail.com',
          start_time='2024-09-17T06:00:00',
          end_time='2024-09-17T12:00:00',
          limit=10
      )
      # Returns up to 10 calendar events between 6:00 AM and 12:00 PM on
      September 17, 2024.

  Args:
      calendar_id (str): the calendar ID to search for events.
      start_time (str): The start of the time range (format is
        YYYY-MM-DDTHH:MM:SS).
      end_time (str): The end of the time range (format is YYYY-MM-DDTHH:MM:SS).
      limit (int): The maximum number of results to return.

  Returns:
      list[dict]: A list of events that match the search criteria.
  """

  creds = Credentials(
      token=credential.oauth2.access_token,
      refresh_token=credential.oauth2.refresh_token,
  )

  service = build("calendar", "v3", credentials=creds)
  events_result = (
      service.events()
      .list(
          calendarId="primary",
          timeMin=start_time + "Z" if start_time else None,
          timeMax=end_time + "Z" if end_time else None,
          maxResults=limit,
          singleEvents=True,
          orderBy="startTime",
      )
      .execute()
  )
  events = events_result.get("items", [])
  return events


def update_time(callback_context: CallbackContext):
  # get current date time
  now = datetime.now()
  formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
  callback_context.state["_time"] = formatted_time


root_agent = Agent(
    model="gemini-2.0-flash",
    name="calendar_agent",
    instruction="""
      You are a helpful personal calendar assistant.
      Use the provided tools to search for calendar events (use 10 as limit if user doesn't specify), and update them.
      Use "primary" as the calendarId if users don't specify.

      Scenario1:
      The user want to query the calendar events.
      Use list_calendar_events to search for calendar events.


      Scenario2:
      User want to know the details of one of the listed calendar events.
      Use google_calendar_events_get to get the details of a calendar event and use redact_event_content to redact confidential information before sending the details to user

      Scenario3:
      User want to update calendar events.
      Use google_calendar_events_update to update calendar events

      IMPORTANT NOTE
      Whenever you use google_calendar_events_get to the details of a calendar event ,
      you MUST use format_calendar_redact_event_content to redact it and use the return value to reply the user.
      This very important! Otherwise you run the risk of leaking confidential information!!!



      Current user:
      <User>
      {userInfo?}
      </User>

      Current time: {_time}
""",
    tools=[
        AuthenticatedFunctionTool(
            func=list_calendar_events,
            auth_config=AuthConfig(
                auth_scheme=OAuth2(
                    flows=OAuthFlows(
                        authorizationCode=OAuthFlowAuthorizationCode(
                            authorizationUrl=(
                                "https://accounts.google.com/o/oauth2/auth"
                            ),
                            tokenUrl="https://oauth2.googleapis.com/token",
                            scopes={
                                "https://www.googleapis.com/auth/calendar": "",
                            },
                        )
                    )
                ),
                raw_auth_credential=AuthCredential(
                    auth_type=AuthCredentialTypes.OAUTH2,
                    oauth2=OAuth2Auth(
                        client_id=oauth_client_id,
                        client_secret=oauth_client_secret,
                    ),
                ),
            ),
        ),
        calendar_toolset,
        redact_event_content,
    ],
    before_agent_callback=update_time,
)
