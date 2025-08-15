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
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from typing import Dict
from typing import Optional
import urllib.parse

from dateutil import parser
from google.genai.errors import ClientError
from tenacity import retry
from tenacity import retry_if_result
from tenacity import RetryError
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from typing_extensions import override

from google import genai

from . import _session_util
from ..events.event import Event
from ..events.event_actions import EventActions
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session

isoparse = parser.isoparse
logger = logging.getLogger('google_adk.' + __name__)


class VertexAiSessionService(BaseSessionService):
  """Connects to the Vertex AI Agent Engine Session Service using GenAI API client.

  https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview
  """

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
  ):
    """Initializes the VertexAiSessionService.

    Args:
      project: The project id of the project to use.
      location: The location of the project to use.
      agent_engine_id: The resource ID of the agent engine to use.
    """
    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id

  async def _get_session_api_response(
      self,
      reasoning_engine_id: str,
      session_id: str,
      api_client: genai.ApiClient,
  ):
    get_session_api_response = await api_client.async_request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )
    get_session_api_response = _convert_api_response(get_session_api_response)
    return get_session_api_response

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    if session_id:
      raise ValueError(
          'User-provided Session id is not supported for'
          ' VertexAISessionService.'
      )
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    api_client = self._get_api_client()

    session_json_dict = {'user_id': user_id}
    if state:
      session_json_dict['session_state'] = state

    api_response = await api_client.async_request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions',
        request_dict=session_json_dict,
    )
    api_response = _convert_api_response(api_response)
    logger.info('Create session response received.')
    logger.debug('Create session response: %s', api_response)

    session_id = api_response['name'].split('/')[-3]
    operation_id = api_response['name'].split('/')[-1]
    if _is_vertex_express_mode(self._project, self._location):
      # Express mode doesn't support LRO, so we need to poll
      # the session resource.
      # TODO: remove this once LRO polling is supported in Express mode.
      @retry(
          stop=stop_after_attempt(5),
          wait=wait_exponential(multiplier=1, min=1, max=3),
          retry=retry_if_result(lambda response: not response),
          reraise=True,
      )
      async def _poll_session_resource():
        try:
          return await self._get_session_api_response(
              reasoning_engine_id, session_id, api_client
          )
        except ClientError:
          logger.info(f'Polling session resource')
          return None

      try:
        await _poll_session_resource()
      except Exception as exc:
        raise ValueError('Failed to create session.') from exc
    else:

      @retry(
          stop=stop_after_attempt(5),
          wait=wait_exponential(multiplier=1, min=1, max=3),
          retry=retry_if_result(
              lambda response: not response.get('done', False),
          ),
          reraise=True,
      )
      async def _poll_lro():
        lro_response = await api_client.async_request(
            http_method='GET',
            path=f'operations/{operation_id}',
            request_dict={},
        )
        lro_response = _convert_api_response(lro_response)
        return lro_response

      try:
        await _poll_lro()
      except RetryError as exc:
        raise TimeoutError(
            f'Timeout waiting for operation {operation_id} to complete.'
        ) from exc
      except Exception as exc:
        raise ValueError('Failed to create session.') from exc

    get_session_api_response = await self._get_session_api_response(
        reasoning_engine_id, session_id, api_client
    )
    session = Session(
        app_name=str(app_name),
        user_id=str(user_id),
        id=str(session_id),
        state=get_session_api_response.get('sessionState', {}),
        last_update_time=isoparse(
            get_session_api_response['updateTime']
        ).timestamp(),
    )
    return session

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    api_client = self._get_api_client()

    # Get session resource
    get_session_api_response = await self._get_session_api_response(
        reasoning_engine_id, session_id, api_client
    )

    if get_session_api_response['userId'] != user_id:
      raise ValueError(f'Session not found: {session_id}')

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

    list_events_api_response = await api_client.async_request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events',
        request_dict={},
    )
    converted_api_response = _convert_api_response(list_events_api_response)

    # Handles empty response case where there are no events to fetch
    if not converted_api_response or converted_api_response.get(
        'httpHeaders', None
    ):
      return session

    session.events += [
        _from_api_event(event)
        for event in converted_api_response['sessionEvents']
    ]

    while converted_api_response.get('nextPageToken', None):
      page_token = converted_api_response.get('nextPageToken', None)
      list_events_api_response = await api_client.async_request(
          http_method='GET',
          path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events?pageToken={page_token}',
          request_dict={},
      )
      converted_api_response = _convert_api_response(list_events_api_response)

      # Handles empty response case where there are no more events to fetch
      if not converted_api_response or converted_api_response.get(
          'httpHeaders', None
      ):
        break
      session.events += [
          _from_api_event(event)
          for event in converted_api_response['sessionEvents']
      ]

    session.events = [
        event for event in session.events if event.timestamp <= update_timestamp
    ]
    session.events.sort(key=lambda event: event.timestamp)

    # Filter events based on config
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
  async def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    api_client = self._get_api_client()

    path = f'reasoningEngines/{reasoning_engine_id}/sessions'
    if user_id:
      parsed_user_id = urllib.parse.quote(f'''"{user_id}"''', safe='')
      path = path + f'?filter=user_id={parsed_user_id}'

    list_sessions_api_response = await api_client.async_request(
        http_method='GET',
        path=path,
        request_dict={},
    )
    list_sessions_api_response = _convert_api_response(
        list_sessions_api_response
    )

    # Handles empty response case
    if not list_sessions_api_response or list_sessions_api_response.get(
        'httpHeaders', None
    ):
      return ListSessionsResponse()

    sessions = []
    for api_session in list_sessions_api_response['sessions']:
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=api_session['name'].split('/')[-1],
          state=api_session.get('sessionState', {}),
          last_update_time=isoparse(api_session['updateTime']).timestamp(),
      )
      sessions.append(session)
    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    api_client = self._get_api_client()

    try:
      await api_client.async_request(
          http_method='DELETE',
          path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
          request_dict={},
      )
    except Exception as e:
      logger.error(f'Error deleting session {session_id}: {e}')
      raise e

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    await super().append_event(session=session, event=event)

    reasoning_engine_id = self._get_reasoning_engine_id(session.app_name)
    api_client = self._get_api_client()
    await api_client.async_request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session.id}:appendEvent',
        request_dict=_convert_event_to_json(event),
    )
    return event

  def _get_reasoning_engine_id(self, app_name: str):
    if self._agent_engine_id:
      return self._agent_engine_id

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

  def _api_client_http_options_override(
      self,
  ) -> Optional[genai.types.HttpOptions]:
    return None

  def _get_api_client(self):
    """Instantiates an API client for the given project and location.

    It needs to be instantiated inside each request so that the event loop
    management can be properly propagated.
    """
    api_client = genai.Client(
        vertexai=True, project=self._project, location=self._location
    )._api_client

    if new_options := self._api_client_http_options_override():
      api_client._http_options = new_options
    return api_client


def _is_vertex_express_mode(
    project: Optional[str], location: Optional[str]
) -> bool:
  """Check if Vertex AI and API key are both enabled replacing project and location, meaning the user is using the Vertex Express Mode."""
  return (
      os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '0').lower() in ['true', '1']
      and os.environ.get('GOOGLE_API_KEY', None) is not None
      and project is None
      and location is None
  )


def _convert_api_response(api_response):
  """Converts the API response to a JSON object based on the type."""
  if hasattr(api_response, 'body'):
    return json.loads(api_response.body)
  return api_response


def _convert_event_to_json(event: Event) -> Dict[str, Any]:
  metadata_json = {
      'partial': event.partial,
      'turn_complete': event.turn_complete,
      'interrupted': event.interrupted,
      'branch': event.branch,
      'custom_metadata': event.custom_metadata,
      'long_running_tool_ids': (
          list(event.long_running_tool_ids)
          if event.long_running_tool_ids
          else None
      ),
  }
  if event.grounding_metadata:
    metadata_json['grounding_metadata'] = event.grounding_metadata.model_dump(
        exclude_none=True, mode='json'
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
    event_json['content'] = event.content.model_dump(
        exclude_none=True, mode='json'
    )
  if event.error_code:
    event_json['error_code'] = event.error_code
  if event.error_message:
    event_json['error_message'] = event.error_message
  return event_json


def _from_api_event(api_event: Dict[str, Any]) -> Event:
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
    event.custom_metadata = api_event['eventMetadata'].get(
        'customMetadata', None
    )
    event.grounding_metadata = _session_util.decode_grounding_metadata(
        api_event['eventMetadata'].get('groundingMetadata', None)
    )
    event.long_running_tool_ids = (
        set(long_running_tool_ids_list) if long_running_tool_ids_list else None
    )

  return event
