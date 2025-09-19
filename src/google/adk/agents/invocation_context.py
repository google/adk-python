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

from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr

from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..plugins.plugin_manager import PluginManager
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.feature_decorator import working_in_progress
from .active_streaming_tool import ActiveStreamingTool
from .base_agent import BaseAgent
from .context_cache_config import ContextCacheConfig
from .live_request_queue import LiveRequestQueue
from .run_config import RunConfig
from .transcription_entry import TranscriptionEntry


class LlmCallsLimitExceededError(Exception):
  """Error thrown when the number of LLM calls exceed the limit."""


class RealtimeCacheEntry(BaseModel):
  """Store audio data chunks for caching before flushing."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )
  """The pydantic model config."""

  role: str
  """The role that created this audio data, typically "user" or "model"."""

  data: types.Blob
  """The audio data chunk."""

  timestamp: float
  """Timestamp when the audio chunk was received."""


class _InvocationCostManager(BaseModel):
  """A container to keep track of the cost of invocation.

  While we don't expect the metrics captured here to be a direct
  representative of monetary cost incurred in executing the current
  invocation, they in some ways have an indirect effect.
  """

  _number_of_llm_calls: int = 0
  """A counter that keeps track of number of llm calls made."""

  def increment_and_enforce_llm_calls_limit(
      self, run_config: Optional[RunConfig]
  ):
    """Increments _number_of_llm_calls and enforces the limit."""
    # We first increment the counter and then check the conditions.
    self._number_of_llm_calls += 1

    if (
        run_config
        and run_config.max_llm_calls > 0
        and self._number_of_llm_calls > run_config.max_llm_calls
    ):
      # We only enforce the limit if the limit is a positive number.
      raise LlmCallsLimitExceededError(
          "Max number of llm calls limit of"
          f" `{run_config.max_llm_calls}` exceeded"
      )


class InvocationContext(BaseModel):
  """An invocation context represents the data of a single invocation of an agent.

  An invocation:
    1. Starts with a user message and ends with a final response.
    2. Can contain one or multiple agent calls.
    3. Is handled by runner.run_async().

  An invocation runs an agent until it does not request to transfer to another
  agent.

  An agent call:
    1. Is handled by agent.run().
    2. Ends when agent.run() ends.

  An LLM agent call is an agent with a BaseLLMFlow.
  An LLM agent call can contain one or multiple steps.

  An LLM agent runs steps in a loop until:
    1. A final response is generated.
    2. The agent transfers to another agent.
    3. The end_invocation is set to true by any callbacks or tools.

  A step:
    1. Calls the LLM only once and yields its response.
    2. Calls the tools and yields their responses if requested.

  The summarization of the function response is considered another step, since
  it is another llm call.
  A step ends when it's done calling llm and tools, or if the end_invocation
  is set to true at any time.

  ```
     ┌─────────────────────── invocation ──────────────────────────┐
     ┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
     ┌──── step_1 ────────┐ ┌───── step_2 ──────┐
     [call_llm] [call_tool] [call_llm] [transfer]
  ```
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )
  """The pydantic model config."""

  artifact_service: Optional[BaseArtifactService] = None
  session_service: BaseSessionService
  memory_service: Optional[BaseMemoryService] = None
  credential_service: Optional[BaseCredentialService] = None
  context_cache_config: Optional[ContextCacheConfig] = None

  invocation_id: str
  """The id of this invocation context. Readonly."""
  branch: Optional[str] = None
  """The branch of the invocation context.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agents shouldn't see their peer agents'
  conversation history.
  """
  agent: BaseAgent
  """The current agent of this invocation context. Readonly."""
  user_content: Optional[types.Content] = None
  """The user content that started this invocation. Readonly."""
  session: Session
  """The current session of this invocation context. Readonly."""

  end_invocation: bool = False
  """Whether to end this invocation.

  Set to True in callbacks or tools to terminate this invocation."""

  live_request_queue: Optional[LiveRequestQueue] = None
  """The queue to receive live requests."""

  active_streaming_tools: Optional[dict[str, ActiveStreamingTool]] = None
  """The running streaming tools of this invocation."""

  transcription_cache: Optional[list[TranscriptionEntry]] = None
  """Caches necessary data, audio or contents, that are needed by transcription."""

  live_session_resumption_handle: Optional[str] = None
  """The handle for live session resumption."""

  input_realtime_cache: Optional[list[RealtimeCacheEntry]] = None
  """Caches input audio chunks before flushing to session and artifact services."""

  output_realtime_cache: Optional[list[RealtimeCacheEntry]] = None
  """Caches output audio chunks before flushing to session and artifact services."""

  run_config: Optional[RunConfig] = None
  """Configurations for live agents under this invocation."""

  plugin_manager: PluginManager = Field(default_factory=PluginManager)
  """The manager for keeping track of plugins in this invocation."""

  _invocation_cost_manager: _InvocationCostManager = PrivateAttr(
      default_factory=_InvocationCostManager
  )
  """A container to keep track of different kinds of costs incurred as a part
  of this invocation.
  """

  def increment_llm_call_count(
      self,
  ):
    """Tracks number of llm calls made.

    Raises:
      LlmCallsLimitExceededError: If number of llm calls made exceed the set
        threshold.
    """
    self._invocation_cost_manager.increment_and_enforce_llm_calls_limit(
        self.run_config
    )

  @property
  def app_name(self) -> str:
    return self.session.app_name

  @property
  def user_id(self) -> str:
    return self.session.user_id

  @working_in_progress("incomplete feature, don't use yet")
  def get_events(
      self,
      current_invocation: bool = False,
      current_branch: bool = False,
  ) -> list[Event]:
    """Returns the events from the current session.

    Args:
      current_invocation: Whether to filter the events by the current
        invocation.
      current_branch: Whether to filter the events by the current branch.

    Returns:
      A list of events from the current session.
    """
    results = self.session.events
    if current_invocation:
      results = [
          event
          for event in results
          if event.invocation_id == self.invocation_id
      ]
    if current_branch:
      results = [event for event in results if event.branch == self.branch]
    return results


def new_invocation_context_id() -> str:
  return "e-" + str(uuid.uuid4())
