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
import json
import os

from dotenv import load_dotenv
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.auth import AuthConfig
from google.adk.auth import AuthCredential
from google.adk.auth import AuthCredentialTypes
from google.adk.auth import OAuth2Auth
from google.adk.tools import ToolContext
from google.adk.tools.google_api_tool import CalendarToolset
from google.auth.transport.requests import Request
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
    tool_filter=["calendar_events_get"],
)


def list_calendar_events(
    start_time: str,
    end_time: str,
    limit: int,
    tool_context: ToolContext,
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
  creds = None

  # Check if the tokes were already in the session state, which means the user
  # has already gone through the OAuth flow and successfully authenticated and
  # authorized the tool to access their calendar.
  if "calendar_tool_tokens" in tool_context.state:
    creds = Credentials.from_authorized_user_info(
        tool_context.state["calendar_tool_tokens"], SCOPES
    )
  if not creds or not creds.valid:
    # If the access token is expired, refresh it with the refresh token.
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      auth_scheme = OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl="https://accounts.google.com/o/oauth2/auth",
                  tokenUrl="https://oauth2.googleapis.com/token",
                  scopes={
                      "https://www.googleapis.com/auth/calendar": (
                          "See, edit, share, and permanently delete all the"
                          " calendars you can access using Google Calendar"
                      )
                  },
              )
          )
      )
      auth_credential = AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id=oauth_client_id, client_secret=oauth_client_secret
          ),
      )
      # If the user has not gone through the OAuth flow before, or the refresh
      # token also expired, we need to ask users to go through the OAuth flow.
      # First we check whether the user has just gone through the OAuth flow and
      # Oauth response is just passed back.
      auth_response = tool_context.get_auth_response(
          AuthConfig(
              auth_scheme=auth_scheme, raw_auth_credential=auth_credential
          )
      )
      if auth_response:
        # ADK exchanged the access token already for us
        access_token = auth_response.oauth2.access_token
        refresh_token = auth_response.oauth2.refresh_token

        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
            client_id=oauth_client_id,
            client_secret=oauth_client_secret,
            scopes=list(auth_scheme.flows.authorizationCode.scopes.keys()),
        )
      else:
        # If there are no auth response which means the user has not gone
        # through the OAuth flow yet, we need to ask users to go through the
        # OAuth flow.
        tool_context.request_credential(
            AuthConfig(
                auth_scheme=auth_scheme,
                raw_auth_credential=auth_credential,
            )
        )
        # The return value is optional and could be any dict object. It will be
        # wrapped in a dict with key as 'result' and value as the return value
        # if the object returned is not a dict. This response will be passed
        # to LLM to generate a user friendly message. e.g. LLM will tell user:
        # "I need your authorization to access your calendar. Please authorize
        # me so I can check your meetings for today."
        return "Need User Authorization to access their calendar."
    # We store the access token and refresh token in the session state for the
    # next runs. This is just an example. On production, a tool should store
    # those credentials in some secure store or properly encrypt it before store
    # it in the session state.
    tool_context.state["calendar_tool_tokens"] = json.loads(creds.to_json())

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
      Use the provided tools to search for calendar events (use 10 as limit if user does't specify), and update them.
      Use "primary" as the calendarId if users don't specify.

      Scenario1:
      The user want to query the calendar events.
      Use list_calendar_events to search for calendar events.


      Scenario2:
      User want to know the details of one of the listed calendar events.
      Use get_calendar_event to get the details of a calendar event.


      Current user:
      <User>
      {userInfo?}
      </User>

      Currnet time: {_time}
""",
    tools=[list_calendar_events, calendar_toolset],
    before_agent_callback=update_time,
)
