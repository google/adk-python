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

"""Utilities for approval tests."""

from typing import Dict, List, Optional, Union

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
from google.adk.sessions.session_service import InMemorySessionService
from google.adk.sessions.state import State


def create_test_session(events=None, state_data=None):
  """Creates a test session with the given events and state.
  
  Args:
    events: A list of events to add to the session.
    state_data: A dictionary of state data to add to the session.
  
  Returns:
    A Session object with the given events and state.
  """
  session_service = InMemorySessionService()
  
  # Create state with the given data
  state = State()
  if state_data:
    for key, value in state_data.items():
      state[key] = value
  
  # Create a session
  session = session_service.create_session(
      app_name="test_app",
      user_id="test_user",
      state=state
  )
  
  # Add events
  if events:
    for event in events:
      # If the event has state_delta, it needs to be processed by append_event
      session_service.append_event(session=session, event=event)
  
  return session
