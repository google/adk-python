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

from typing import Any
from typing import cast
from typing import Optional

from google.adk.platform import time as platform_time
from google.adk.platform import uuid as platform_uuid
from google.genai import types
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from pydantic import PrivateAttr

from ..models.llm_response import LlmResponse
from .event_actions import EventActions


class NodeInfo(BaseModel):
  """Workflow node metadata attached to an Event."""

  model_config = ConfigDict(
      ser_json_bytes='base64',
      val_json_bytes='base64',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  path: str = ''
  """The path of the node in the workflow.
  In a workflow A, if node B is directly under A, and B emits an event, the
  path will be "A/B". Agent state event will have path as "A".
  """

  output_for: list[str] | None = None
  """Node paths whose output this event represents.

  Set on events that carry an output value. When set, the output field
  of this event is also considered the output for each listed node path
  in the same invocation. For example, ``["wf/A@1/B@1", "wf/A@1"]`` means
  this event's output counts as the output for both.
  """

  message_as_output: bool | None = None
  """When True, this event's content is the node's output.

  No separate output event is needed — the content event already
  carries the output value.
  """

  @property
  def run_id(self) -> str:
    """The run ID of the node that generated the event."""
    from ._node_path_builder import _NodePathBuilder

    return _NodePathBuilder.from_string(self.path).run_id or ''

  @property
  def parent_run_id(self) -> str | None:
    """The run ID of the parent node that dynamically scheduled
    this node. Used to reconstruct dynamic node state from session events."""
    from ._node_path_builder import _NodePathBuilder

    builder = _NodePathBuilder.from_string(self.path)
    if builder.parent:
      return builder.parent.run_id
    return None

  @property
  def name(self) -> str:
    """The clean name of the node (without @run_id)."""
    from ._node_path_builder import _NodePathBuilder

    return _NodePathBuilder.from_string(self.path).node_name


class Event(LlmResponse):
  """Represents an event in a conversation between agents and users.

  It is used to store the content of the conversation, as well as the actions
  taken by the agents like function calls, etc.
  """

  model_config = ConfigDict(
      extra='ignore',
      ser_json_bytes='base64',
      val_json_bytes='base64',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  invocation_id: str = ''
  """The invocation ID of the event. Should be non-empty before appending to a session."""
  author: str = ''
  """'user' or the name of the agent, indicating who appended the event to the
  session."""
  actions: EventActions = Field(default_factory=EventActions)
  """The actions taken by the agent."""

  output: Any | None = None
  """Generic data output from a workflow node."""

  node_info: NodeInfo = Field(default_factory=NodeInfo)
  """Workflow node metadata (path, run_id, etc.)."""

  long_running_tool_ids: set[str] | None = None
  """Set of ids of the long running function calls.
  Agent client will know from this field about which function call is long running.
  only valid for function call event
  """
  branch: str | None = None
  """The branch of the event.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agent shouldn't see their peer agents'
  conversation history.
  """
  isolation_scope: str | None = None
  """Scope tag indicating which logical context this event belongs to.

  When set, the LLM content-builder restricts session events visible to
  an agent to those whose ``isolation_scope`` matches the agent's own
  scope.  One usage today is the Task API: a delegated task agent is
  scoped under the originating function-call id (``<fc_id>``) so it
  sees only its own task's events, isolated from the chat
  coordinator's broader conversation.

  ⚠️ DO NOT USE THIS FIELD DIRECTLY.  It is an internal mechanism that
  may change without notice.  External code should not read, write, or
  rely on its semantics.
  """

  # The following are computed fields.
  # Do not assign the ID. It will be assigned by the session.
  id: str = ''
  """The unique identifier of the event."""
  timestamp: float = Field(default_factory=lambda: platform_time.get_time())
  """The timestamp of the event."""

  @model_validator(mode='before')
  @classmethod
  def _accept_convenience_kwargs(cls, data: Any) -> Any:
    """Routes convenience kwargs to nested fields.

    Routed kwargs:
      message: ContentUnion -> content (converted via t_content)
      state: dict           -> actions.state_delta
      route: value          -> actions.route
      node_path: str        -> node_info.path

    Subclasses that declare any of these as real fields (or aliases of
    real fields) keep normal field validation behavior.
    """
    if not isinstance(data, dict):
      return data

    field_names: set[str] = set(cls.model_fields.keys())
    for f in cls.model_fields.values():
      if f.alias:
        field_names.add(f.alias)
    message = None if 'message' in field_names else data.pop('message', None)
    state = None if 'state' in field_names else data.pop('state', None)
    route = None if 'route' in field_names else data.pop('route', None)
    node_path = (
        None if 'node_path' in field_names else data.pop('node_path', None)
    )

    if message is not None:
      if data.get('content') is not None:
        raise ValueError(
            "'message' and 'content' are mutually exclusive."
            ' Use one or the other.'
        )
      from google.genai import _transformers

      data['content'] = _transformers.t_content(message)

    if state is not None or route is not None:
      actions = data.get('actions')
      actions_dict: Optional[dict[str, Any]] = None
      if actions is None:
        actions_dict = {}
      elif isinstance(actions, EventActions):
        actions_dict = actions.model_dump()
      elif isinstance(actions, dict):
        actions_dict = dict(actions)
      # If actions is an unexpected type, skip the transformation and let
      # Pydantic's normal field validation report the error.
      if actions_dict is not None:
        if state is not None:
          actions_dict['state_delta'] = state
        if route is not None:
          actions_dict['route'] = route
        data['actions'] = actions_dict

    if node_path is not None:
      node_info = data.get('node_info')
      node_info_dict: Optional[dict[str, Any]] = None
      if node_info is None:
        node_info_dict = {}
      elif isinstance(node_info, NodeInfo):
        node_info_dict = node_info.model_dump()
      elif isinstance(node_info, dict):
        node_info_dict = dict(node_info)
      # If node_info is an unexpected type, skip the transformation and let
      # Pydantic's normal field validation report the error.
      if node_info_dict is not None:
        node_info_dict['path'] = node_path
        data['node_info'] = node_info_dict

    return data

  @property
  def message(self) -> Optional[types.Content]:
    """Alias for content. Returns the user-facing message of the event."""
    return self.content

  @message.setter
  def message(self, value: Optional[types.ContentUnion]) -> None:
    """Sets the content of the event."""
    if value is not None:
      from google.genai import _transformers

      self.content = _transformers.t_content(value)
    else:
      self.content = None

  @property
  def node_name(self) -> str:
    """The name of the node that generated the event."""
    if self.actions.agent_state or self.actions.end_of_agent:
      return ''
    return self.node_info.name

  def model_post_init(self, __context):
    """Post initialization logic for the event."""
    # Generates a random ID for the event.
    if not self.id:
      self.id = Event.new_id()

  def is_final_response(self) -> bool:
    """Returns whether the event is the final response of an agent.

    NOTE: This method is ONLY for use by Agent Development Kit.

    Note that when multiple agents participate in one invocation, there could be
    one event has `is_final_response()` as True for each participating agent.
    """
    if self.actions.skip_summarization or self.long_running_tool_ids:
      return True
    return (
        not self.get_function_calls()
        and not self.get_function_responses()
        and not self.partial
        and not self.has_trailing_code_execution_result()
    )

  def has_trailing_code_execution_result(
      self,
  ) -> bool:
    """Returns whether the event has a trailing code execution result."""
    if self.content:
      if self.content.parts:
        return self.content.parts[-1].code_execution_result is not None
    return False

  @staticmethod
  def new_id() -> str:
    return cast(str, platform_uuid.new_uuid())
