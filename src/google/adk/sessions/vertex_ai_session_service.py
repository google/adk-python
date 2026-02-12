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
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from google.genai.errors import ClientError
from typing_extensions import override

if TYPE_CHECKING:
  import vertexai

from . import _session_util
from ..events.event import Event
from ..events.event_actions import EventActions
from ..utils.vertex_ai_utils import get_express_mode_api_key
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session

logger = logging.getLogger('google_adk.' + __name__)


class VertexAiSessionService(BaseSessionService):
  """Connects to the Vertex AI Agent Engine Session Service using Agent Engine SDK.

  https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview
  """

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
      *,
      express_mode_api_key: Optional[str] = None,
      agents_dir: Optional[str] = None,
  ):
    """Initializes the VertexAiSessionService.

    Args:
      project: The project id of the project to use. If not provided, will be
        resolved lazily at runtime from environment variables or .env files.
      location: The location of the project to use. If not provided, will be
        resolved lazily at runtime from environment variables or .env files.
      agent_engine_id: The resource ID of the agent engine to use.
      express_mode_api_key: The API key to use for Express Mode. If not
        provided, the API key from the GOOGLE_API_KEY environment variable will
        be used. It will only be used if GOOGLE_GENAI_USE_VERTEXAI is true.
        Do not use Google AI Studio API key for this field. For more details,
        visit
        https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
      agents_dir: The directory containing agent configurations and .env files.
        Used for lazy resolution of project/location when not explicitly provided.
    """
    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id
    self._agents_dir = agents_dir
    self._config_resolved = False
    self._express_mode_api_key = express_mode_api_key

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    """Creates a new session.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      state: The initial state of the session.
      session_id: The ID of the session.
      **kwargs: Additional arguments to pass to the session creation. E.g. set
        expire_time='2025-10-01T00:00:00Z' to set the session expiration time.
        See https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.reasoningEngines.sessions
        for more details.
    Returns:
      The created session.
    """
    # Lazily resolve project/location on first use
    self._resolve_config(app_name)

    if session_id:
      raise ValueError(
          'User-provided Session id is not supported for'
          ' VertexAISessionService.'
      )

    reasoning_engine_id = self._get_reasoning_engine_id(app_name)

    config = {'session_state': state} if state else {}
    config.update(kwargs)
    async with self._get_api_client() as api_client:
      api_response = await api_client.agent_engines.sessions.create(
          name=f'reasoningEngines/{reasoning_engine_id}',
          user_id=user_id,
          config=config,
      )
      logger.debug('Create session response: %s', api_response)
      get_session_response = api_response.response
      session_id = get_session_response.name.split('/')[-1]

    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=getattr(get_session_response, 'session_state', None) or {},
        last_update_time=get_session_response.update_time.timestamp(),
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
    # Lazily resolve project/location on first use
    self._resolve_config(app_name)
    
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    session_resource_name = (
        f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}'
    )
    async with self._get_api_client() as api_client:
      # Get session resource and events in parallel.
      list_events_kwargs = {}
      if config and not config.num_recent_events and config.after_timestamp:
        # Filter events based on timestamp.
        list_events_kwargs['config'] = {
            'filter': 'timestamp>="{}"'.format(
                datetime.datetime.fromtimestamp(
                    config.after_timestamp, tz=datetime.timezone.utc
                ).isoformat()
            )
        }

      try:
        get_session_response, events_iterator = await asyncio.gather(
            api_client.agent_engines.sessions.get(name=session_resource_name),
            api_client.agent_engines.sessions.events.list(
                name=session_resource_name,
                **list_events_kwargs,
            ),
        )
      except ClientError as e:
        if e.code == 404:
          logger.debug(
              'Session %s not found in Vertex AI Agent Engine.',
              session_resource_name,
          )
          return None
        raise
      if get_session_response.user_id != user_id:
        raise ValueError(
            f'Session {session_id} does not belong to user {user_id}.'
        )

      update_timestamp = get_session_response.update_time.timestamp()
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=getattr(get_session_response, 'session_state', None) or {},
          last_update_time=update_timestamp,
      )
      # Preserve the entire event stream that Vertex returns rather than trying
      # to discard events written milliseconds after the session resource was
      # updated. Clock skew between those writes can otherwise drop tool_result
      # events and permanently break the replayed conversation.
      async for event in events_iterator:
        session.events.append(_from_api_event(event))

    if config:
      # Filter events based on num_recent_events.
      if config.num_recent_events:
        session.events = session.events[-config.num_recent_events :]

    return session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    # Lazily resolve project/location on first use
    self._resolve_config(app_name)
    
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)

    async with self._get_api_client() as api_client:
      sessions = []
      config = {}
      if user_id is not None:
        config['filter'] = f'user_id="{user_id}"'
      sessions_iterator = await api_client.agent_engines.sessions.list(
          name=f'reasoningEngines/{reasoning_engine_id}',
          config=config,
      )

      async for api_session in sessions_iterator:
        sessions.append(
            Session(
                app_name=app_name,
                user_id=api_session.user_id,
                id=api_session.name.split('/')[-1],
                state=getattr(api_session, 'session_state', None) or {},
                last_update_time=api_session.update_time.timestamp(),
            )
        )

    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    # Lazily resolve project/location on first use
    self._resolve_config(app_name)
    
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)

    async with self._get_api_client() as api_client:
      try:
        await api_client.agent_engines.sessions.delete(
            name=(
                f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}'
            ),
        )
      except Exception as e:
        logger.error('Error deleting session %s: %s', session_id, e)
        raise

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    await super().append_event(session=session, event=event)

    # Lazily resolve project/location on first use
    self._resolve_config(session.app_name)

    reasoning_engine_id = self._get_reasoning_engine_id(session.app_name)

    config = {}
    if event.content:
      config['content'] = event.content.model_dump(
          exclude_none=True, mode='json'
      )
    if event.actions:
      config['actions'] = {
          'skip_summarization': event.actions.skip_summarization,
          'state_delta': event.actions.state_delta,
          'artifact_delta': event.actions.artifact_delta,
          'transfer_agent': event.actions.transfer_to_agent,
          'escalate': event.actions.escalate,
          'requested_auth_configs': {
              k: json.loads(v.model_dump_json(exclude_none=True, by_alias=True))
              for k, v in event.actions.requested_auth_configs.items()
          },
          # TODO: add requested_tool_confirmations, compaction, agent_state once
          # they are available in the API.
      }
    if event.error_code:
      config['error_code'] = event.error_code
    if event.error_message:
      config['error_message'] = event.error_message

    metadata_dict = {
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
      metadata_dict['grounding_metadata'] = event.grounding_metadata.model_dump(
          exclude_none=True, mode='json'
      )
    config['event_metadata'] = metadata_dict

    async with self._get_api_client() as api_client:
      await api_client.agent_engines.sessions.events.append(
          name=f'reasoningEngines/{reasoning_engine_id}/sessions/{session.id}',
          author=event.author,
          invocation_id=event.invocation_id,
          timestamp=datetime.datetime.fromtimestamp(
              event.timestamp, tz=datetime.timezone.utc
          ),
          config=config,
      )
    return event

  def _get_reasoning_engine_id(self, app_name: str):
    if self._agent_engine_id:
      return self._agent_engine_id

    if app_name.isdigit():
      return app_name

    pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$'
    match = re.fullmatch(pattern, app_name)

    if not match:
      raise ValueError(
          f'App name {app_name} is not valid. It should either be the full'
          ' ReasoningEngine resource name, or the reasoning engine id.'
      )

    return match.groups()[-1]

  def _resolve_config(self, app_name: Optional[str] = None) -> None:
    """Lazily resolves project and location if not provided at initialization.
    
    This method is called on first use to resolve GCP configuration from:
    1. Explicit environment variables (highest priority)
    2. Agent-specific .env file (if app_name and agents_dir provided)
    3. agents_dir root .env file
    4. Parent directory .env files (walking upward)
    
    Args:
      app_name: Optional app name to load agent-specific .env files.
    
    Raises:
      ValueError: If project or location cannot be resolved.
    """
    if self._config_resolved:
      return
    
    import os
    
    # If both are already set (either at init or via environment), we're done
    if self._project and self._location:
      self._config_resolved = True
      self._express_mode_api_key = get_express_mode_api_key(
          self._project, self._location, self._express_mode_api_key
      )
      return
    
    # Try to load from environment and .env files
    if self._agents_dir and app_name:
      # Load agent-specific .env
      from ..cli.utils import envs
      envs.load_dotenv_for_agent(app_name, self._agents_dir)
    elif self._agents_dir:
      # Load from agents_dir root
      from ..cli.utils import envs
      envs.load_dotenv_for_agent("", self._agents_dir)
    
    # Resolve from environment after loading .env
    self._project = self._project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self._location = self._location or os.environ.get("GOOGLE_CLOUD_LOCATION")
    
    if not self._project or not self._location:
      error_msg = (
          "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set. "
          "You can set them via:\n"
          "  1. Environment variables: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION\n"
          "  2. .env file in your agent directory\n"
          "  3. .env file in your agents_dir\n"
          "  4. Use full resource name: agentengine://projects/{project}/locations/{location}/reasoningEngines/{id}"
      )
      raise ValueError(error_msg)
    
    self._config_resolved = True
    self._express_mode_api_key = get_express_mode_api_key(
        self._project, self._location, self._express_mode_api_key
    )

  def _api_client_http_options_override(
      self,
  ) -> Optional[Union[types.HttpOptions, types.HttpOptionsDict]]:
    return None

  def _get_api_client(self) -> vertexai.AsyncClient:
    """Instantiates an API client for the given project and location.

    Returns:
      An API client for the given project and location or express mode api key.
    """
    import vertexai

    return vertexai.Client(
        project=self._project,
        location=self._location,
        http_options=self._api_client_http_options_override(),
        api_key=self._express_mode_api_key,
    ).aio


def _from_api_event(api_event_obj: vertexai.types.SessionEvent) -> Event:
  """Converts an API event object to an Event object."""
  actions = getattr(api_event_obj, 'actions', None)
  if actions:
    actions_dict = actions.model_dump(exclude_none=True, mode='python')
    rename_map = {'transfer_agent': 'transfer_to_agent'}
    renamed_actions_dict = {
        rename_map.get(k, k): v for k, v in actions_dict.items()
    }
    event_actions = EventActions.model_validate(renamed_actions_dict)
  else:
    event_actions = EventActions()

  event_metadata = getattr(api_event_obj, 'event_metadata', None)
  if event_metadata:
    long_running_tool_ids_list = getattr(
        event_metadata, 'long_running_tool_ids', None
    )
    long_running_tool_ids = (
        set(long_running_tool_ids_list) if long_running_tool_ids_list else None
    )
    partial = getattr(event_metadata, 'partial', None)
    turn_complete = getattr(event_metadata, 'turn_complete', None)
    interrupted = getattr(event_metadata, 'interrupted', None)
    branch = getattr(event_metadata, 'branch', None)
    custom_metadata = getattr(event_metadata, 'custom_metadata', None)
    grounding_metadata = _session_util.decode_model(
        getattr(event_metadata, 'grounding_metadata', None),
        types.GroundingMetadata,
    )
  else:
    long_running_tool_ids = None
    partial = None
    turn_complete = None
    interrupted = None
    branch = None
    custom_metadata = None
    grounding_metadata = None

  return Event(
      id=api_event_obj.name.split('/')[-1],
      invocation_id=api_event_obj.invocation_id,
      author=api_event_obj.author,
      actions=event_actions,
      content=_session_util.decode_model(
          getattr(api_event_obj, 'content', None), types.Content
      ),
      timestamp=api_event_obj.timestamp.timestamp(),
      error_code=getattr(api_event_obj, 'error_code', None),
      error_message=getattr(api_event_obj, 'error_message', None),
      partial=partial,
      turn_complete=turn_complete,
      interrupted=interrupted,
      branch=branch,
      custom_metadata=custom_metadata,
      grounding_metadata=grounding_metadata,
      long_running_tool_ids=long_running_tool_ids,
  )
