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
import re
import time
from typing import Any, Optional

from dateutil import parser
from google import genai
from typing_extensions import override

from ..events.event import Event
from ..events.event_actions import EventActions
from . import _session_util
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListEventsResponse
from .base_session_service import ListSessionsResponse
from .session import Session


isoparse = parser.isoparse
logger = logging.getLogger(__name__)


class VertexAiSessionService(BaseSessionService):
  """Connects to the managed Vertex AI Session Service."""

  def __init__(
      self,
      project: str = None,
      location: str = None,
  ):
    self.project = project
    self.location = location

    client = genai.Client(vertexai=True, project=project, location=location)
    self.api_client = client._api_client

  @override
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    session_json_dict = {'user_id': user_id}
    if state:
      session_json_dict['session_state'] = state

    api_response = self.api_client.request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions',
        request_dict=session_json_dict,
    )
    logger.info(f'Create Session response {api_response}')

    session_id = api_response['name'].split('/')[-3]
    operation_id = api_response['name'].split('/')[-1]

    max_retry_attempt = 5
    while max_retry_attempt >= 0:
      lro_response = self.api_client.request(
          http_method='GET',
          path=f'operations/{operation_id}',
          request_dict={},
      )

      if lro_response.get('done', None):
        break

      time.sleep(1)
      max_retry_attempt -= 1

    # Get session resource
    get_session_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

    update_timestamp = isoparse(
        get_session_api_response['updateTime']
    ).timestamp()
    session = Session(
        app_name=str(app_name),
        user_id=str(user_id),
        id=str(session_id),
        state=get_session_api_response.get('sessionState', {}),
        last_update_time=update_timestamp,
    )
    return session

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Session:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    # Get session resource
    get_session_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

    session_id = get_session_api_response['name'].split('/')[-1]
    update_timestamp = isoparse(
        get_session_api_response['updateTime']
    ).timestamp()
    session = Session(
        app_name=str(app_name),
        user_id=str(user_id),
        id=str(session_id),
        state=get_session_api_response.get('sessionState', {}),
        last_update_time=update_timestamp,
    )

    list_events_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events',
        request_dict={},
    )

    # Handles empty response case
    if list_events_api_response.get('httpHeaders', None):
      return session

    session.events = [
        _from_api_event(event)
        for event in list_events_api_response['sessionEvents']
    ]
    session.events = [
        event for event in session.events if event.timestamp <= update_timestamp
    ]
    session.events.sort(key=lambda event: event.timestamp)

    if config:
      if config.num_recent_events:
        session.events = session.events[-config.num_recent_events :]
      elif config.after_timestamp:
        i = len(session.events) - 1
        while i >= 0:
          if session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          session.events = session.events[i:]

    return session

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions?filter=user_id={user_id}',
        request_dict={},
    )

    # Handles empty response case
    if api_response.get('httpHeaders', None):
      return ListSessionsResponse()

    sessions = []
    for api_session in api_response['sessions']:
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=api_session['name'].split('/')[-1],
          state={},
          last_update_time=isoparse(api_session['updateTime']).timestamp(),
      )
      sessions.append(session)
    return ListSessionsResponse(sessions=sessions)

  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)
    self.api_client.request(
        http_method='DELETE',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

  @override
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)
    api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events',
        request_dict={},
    )

    logger.info(f'List events response {api_response}')

    # Handles empty response case
    if api_response.get('httpHeaders', None):
      return ListEventsResponse()

    session_events = api_response['sessionEvents']

    return ListEventsResponse(
        events=[_from_api_event(event) for event in session_events]
    )

  @override
  def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    super().append_event(session=session, event=event)

    reasoning_engine_id = _parse_reasoning_engine_id(session.app_name)
    self.api_client.request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session.id}:appendEvent',
        request_dict=_convert_event_to_json(event),
    )

    return event


def _convert_event_to_json(event: Event):
  metadata_json = {
      'partial': event.partial,
      'turn_complete': event.turn_complete,
      'interrupted': event.interrupted,
      'branch': event.branch,
      'long_running_tool_ids': (
          list(event.long_running_tool_ids)
          if event.long_running_tool_ids
          else None
      ),
  }
  if event.grounding_metadata:
    metadata_json['grounding_metadata'] = event.grounding_metadata.model_dump(
        exclude_none=True
    )

  event_json = {
      'author': event.author,
      'invocation_id': event.invocation_id,
      'timestamp': {
          'seconds': int(event.timestamp),
          'nanos': int(
              (event.timestamp - int(event.timestamp)) * 1_000_000_000
          ),
      },
      'error_code': event.error_code,
      'error_message': event.error_message,
      'event_metadata': metadata_json,
  }

  if event.actions:
    actions_json = {
        'skip_summarization': event.actions.skip_summarization,
        'state_delta': event.actions.state_delta,
        'artifact_delta': event.actions.artifact_delta,
        'transfer_agent': event.actions.transfer_to_agent,
        'escalate': event.actions.escalate,
        'requested_auth_configs': event.actions.requested_auth_configs,
    }
    event_json['actions'] = actions_json
  if event.content:
    event_json['content'] = _session_util.encode_content(event.content)
  if event.error_code:
    event_json['error_code'] = event.error_code
  if event.error_message:
    event_json['error_message'] = event.error_message
  return event_json


def _from_api_event(api_event: dict) -> Event:
  event_actions = EventActions()
  if api_event.get('actions', None):
    event_actions = EventActions(
        skip_summarization=api_event['actions'].get('skipSummarization', None),
        state_delta=api_event['actions'].get('stateDelta', {}),
        artifact_delta=api_event['actions'].get('artifactDelta', {}),
        transfer_to_agent=api_event['actions'].get('transferAgent', None),
        escalate=api_event['actions'].get('escalate', None),
        requested_auth_configs=api_event['actions'].get(
            'requestedAuthConfigs', {}
        ),
    )

  event = Event(
      id=api_event['name'].split('/')[-1],
      invocation_id=api_event['invocationId'],
      author=api_event['author'],
      actions=event_actions,
      content=_session_util.decode_content(api_event.get('content', None)),
      timestamp=isoparse(api_event['timestamp']).timestamp(),
      error_code=api_event.get('errorCode', None),
      error_message=api_event.get('errorMessage', None),
  )

  if api_event.get('eventMetadata', None):
    long_running_tool_ids_list = api_event['eventMetadata'].get(
        'longRunningToolIds', None
    )
    event.partial = api_event['eventMetadata'].get('partial', None)
    event.turn_complete = api_event['eventMetadata'].get('turnComplete', None)
    event.interrupted = api_event['eventMetadata'].get('interrupted', None)
    event.branch = api_event['eventMetadata'].get('branch', None)
    event.grounding_metadata = api_event['eventMetadata'].get(
        'groundingMetadata', None
    )
    event.long_running_tool_ids = (
        set(long_running_tool_ids_list) if long_running_tool_ids_list else None
    )

  return event


def _parse_reasoning_engine_id(app_name: str):
  if app_name.isdigit():
    return app_name

  pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$'
  match = re.fullmatch(pattern, app_name)

  if not bool(match):
    raise ValueError(
        f'App name {app_name} is not valid. It should either be the full'
        ' ReasoningEngine resource name, or the reasoning engine id.'
    )

  return match.groups()[-1]
