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
from typing import Any

from google.adk.platform import uuid as platform_uuid
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..apps._configs import EventsCompactionConfig
from ..apps._configs import ResumabilityConfig
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.auth_credential import AuthCredential
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..events._branch_path import _BranchPath
from ..events.event import Event
from ..live._audio_cache_manager import RealtimeCacheEntry as RealtimeCacheEntry
from ..live.live_request_queue import LiveRequestQueue
from ..memory.base_memory_service import BaseMemoryService
from ..plugins.plugin_manager import PluginManager
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..tools.base_tool import BaseTool
from ..workflow._base_node import BaseNode
from .active_streaming_tool import ActiveStreamingTool
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .context_cache_config import ContextCacheConfig
from .run_config import RunConfig
from .transcription_entry import TranscriptionEntry

_EventQueueItem = tuple[object, asyncio.Event | None]


class LlmCallsLimitExceededError(Exception):
  """Error thrown when the number of LLM calls exceed the limit."""


class _InvocationCostManager(BaseModel):
  """A container to keep track of the cost of invocation.

  While we don't expect the metrics captured here to be a direct
  representative of monetary cost incurred in executing the current
  invocation, they in some ways have an indirect effect.
  """

  _number_of_llm_calls: int = 0
  """A counter that keeps track of number of llm calls made."""

  def increment_and_enforce_llm_calls_limit(
      self, run_config: RunConfig | None
  ) -> None:
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

  artifact_service: BaseArtifactService | None = None
  session_service: BaseSessionService
  memory_service: BaseMemoryService | None = None
  credential_service: BaseCredentialService | None = None
  context_cache_config: ContextCacheConfig | None = None

  invocation_id: str
  """The id of this invocation context. Readonly."""
  branch: str | None = None
  """The branch of the invocation context.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agents shouldn't see their peer agents'
  conversation history.
  """
  isolation_scope: str | None = None
  """Scope tag for filtering session events visible to this agent.

  When set, the LLM content-builder restricts session events to those
  whose ``event.isolation_scope`` matches.  One usage today is the
  Task API: task-mode and single_turn-mode agents are scoped under
  the originating function-call id; chat coordinators are unscoped
  and see only unscoped events.

  ⚠️ DO NOT USE THIS FIELD DIRECTLY.  It is an internal mechanism
  that may change without notice.
  """
  agent: BaseAgent | BaseNode | None = None
  """The current agent of this invocation context.

  None when Runner drives a BaseNode (not a BaseAgent).
  """
  user_content: types.Content | None = None
  """The user content that started this invocation. Readonly."""
  session: Session
  """The current session of this invocation context. Readonly."""

  node_path: str | None = None
  """The path of the current agent in the workflow call stack.

  Used by workflow agents to track their position in nested agent hierarchies.
  Format: "agent_1/agent_2/agent_3" where agent_1 is the outermost workflow.
  None for non-workflow agents.
  """

  agent_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
  """The state of the agent for this invocation."""

  end_of_agents: dict[str, bool] = Field(default_factory=dict)
  """The end of agent status for each agent in this invocation."""

  end_invocation: bool = False
  """Whether to end this invocation.

  Set to True in callbacks or tools to terminate this invocation."""

  live_request_queue: LiveRequestQueue | None = None
  """The queue to receive live requests."""

  active_streaming_tools: dict[str, ActiveStreamingTool] | None = None
  """The running streaming tools of this invocation."""

  active_non_blocking_tool_tasks: dict[str, asyncio.Task[Any]] | None = None
  """The running non-blocking tool tasks of this invocation (Live only)."""

  transcription_cache: list[TranscriptionEntry] | None = None
  """Caches necessary data, audio or contents, that are needed by transcription."""

  live_session_resumption_handle: str | None = None
  """The handle for live session resumption."""

  input_realtime_cache: list[RealtimeCacheEntry] | None = None
  """Caches input audio chunks before flushing to session and artifact services."""

  output_realtime_cache: list[RealtimeCacheEntry] | None = None
  """Caches output audio chunks before flushing to session and artifact services."""

  run_config: RunConfig | None = None
  """Configurations for live agents under this invocation."""

  resumability_config: ResumabilityConfig | None = None
  """The resumability config that applies to all agents under this invocation."""

  events_compaction_config: EventsCompactionConfig | None = None
  """The compaction config for this invocation."""

  token_compaction_checked: bool = False
  """Whether token-threshold compaction ran during this invocation."""

  plugin_manager: PluginManager = Field(default_factory=PluginManager)
  """The manager for keeping track of plugins in this invocation."""

  _state_schema: type[BaseModel] | None = None
  """The Pydantic model declaring the expected state keys and types.

  Propagated from the owning agent down the hierarchy.  When set,
  ``ctx.state`` mutations and ``Event(state={...})`` deltas are
  validated against this schema at runtime.
  """

  canonical_tools_cache: list[BaseTool] | None = None
  """The cache of canonical tools for this invocation."""

  _event_queue: asyncio.Queue[_EventQueueItem] | None = PrivateAttr(
      default=None
  )
  """Shared event queue for all nodes in this invocation.

  All nodes enqueue events here via ``_enqueue_event()``. The Runner
  main loop is the sole consumer — it appends events to session and
  yields them to SSE.
  """

  credential_by_key: dict[str, AuthCredential] = Field(default_factory=dict)
  """The resolved credentials for this invocation, keyed by credential_key."""

  _custom_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
  """Custom metadata for attaching low-level execution telemetry."""

  _invocation_cost_manager: _InvocationCostManager = PrivateAttr(
      default_factory=_InvocationCostManager
  )
  """A container to keep track of different kinds of costs incurred as a part
  of this invocation.
  """

  @override
  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    if self.run_config and self.run_config.custom_metadata:
      self._custom_metadata.update(self.run_config.custom_metadata)

  @property
  def is_resumable(self) -> bool:
    """Returns whether the current invocation is resumable."""
    return (
        self.resumability_config is not None
        and self.resumability_config.is_resumable
    )

  async def _enqueue_event(self, event: Event) -> None:
    """Enqueue an event for the Runner main loop to process.

    Non-partial events block until the main loop has appended them
    to session, ensuring session consistency before the node
    continues. Partial events (SSE streaming) flow through without
    blocking.
    """
    if self._event_queue is None:
      raise RuntimeError(
          "_enqueue_event called but _event_queue is not set. "
          "Ensure the Runner initialises _event_queue on "
          "InvocationContext."
      )

    if event.partial:
      # Partial events: SSE streaming only, no session append, no blocking.
      await self._event_queue.put((event, None))
    else:
      # Non-partial events: block until main loop appends to session.
      processed = asyncio.Event()
      await self._event_queue.put((event, processed))
      await processed.wait()

  def set_agent_state(
      self,
      agent_name: str,
      *,
      agent_state: BaseAgentState | None = None,
      end_of_agent: bool = False,
  ) -> None:
    """Sets the state of an agent in this invocation.

    * If end_of_agent is True, will set the end_of_agent flag to True and
      clear the agent_state.
    * Otherwise, if agent_state is not None, will set the agent_state and
      reset the end_of_agent flag to False.
    * Otherwise, will clear the agent_state and end_of_agent flag, to allow the
      agent to re-run.

    Args:
      agent_name: The name of the agent.
      agent_state: The state of the agent. Will be ignored if end_of_agent is
        True.
      end_of_agent: Whether the agent has finished running.
    """
    if end_of_agent:
      self.end_of_agents[agent_name] = True
      self.agent_states.pop(agent_name, None)
    elif agent_state is not None:
      self.agent_states[agent_name] = agent_state.model_dump(mode="json")
      self.end_of_agents[agent_name] = False
    else:
      self.end_of_agents.pop(agent_name, None)
      self.agent_states.pop(agent_name, None)

  def reset_sub_agent_states(
      self,
      agent_name: str,
  ) -> None:
    """Resets the state of all sub-agents of the given agent in this invocation.

    Args:
      agent_name: The name of the agent whose sub-agent states need to be reset.
    """
    if not isinstance(self.agent, BaseAgent):
      return
    agent = self.agent.find_agent(agent_name)
    if not agent:
      return

    for sub_agent in agent.sub_agents:
      # Reset the sub-agent's state in the context to ensure that each
      # sub-agent starts fresh.
      self.set_agent_state(sub_agent.name)
      self.reset_sub_agent_states(sub_agent.name)

  def populate_invocation_agent_states(self) -> None:
    """Populates agent states for the current invocation if it is resumable.

    For history events that contain agent state information, set the
    agent_state and end_of_agent of the agent that generated the event.

    For non-workflow agents, also set an initial agent_state if it has
    already generated some contents.
    """
    if not self.is_resumable:
      return
    for event in self._get_events(current_invocation=True):
      # Use node_info.path if available (workflow events), otherwise fall
      # back to author (non-workflow events).
      key = event.node_info.path or event.author
      if event.actions.end_of_agent:
        self.end_of_agents[key] = True
        # Delete agent_state when it is end
        self.agent_states.pop(key, None)
      elif event.actions.agent_state is not None:
        self.agent_states[key] = event.actions.agent_state
        # Invalidate the end_of_agent flag
        self.end_of_agents[key] = False
      elif (
          event.author != "user"
          and event.content
          and not self.agent_states.get(key)
      ):
        # If the agent has generated some contents but its agent_state is not
        # set, set its agent_state to an empty agent_state.
        self.agent_states[key] = BaseAgentState().model_dump(mode="json")
        # Invalidate the end_of_agent flag
        self.end_of_agents[key] = False

  def increment_llm_call_count(
      self,
  ) -> None:
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

  # TODO: Move this method from invocation_context to a dedicated module.
  def _get_events(
      self,
      *,
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
      branch_fc_ids: set[str] | None = None

      def _branch_function_call_ids() -> set[str]:
        """Function call ids issued on this branch or a descendant sub-branch.

        The ids depend only on the session and this branch, not on the event
        being tested, so they are gathered at most once per call. Gathering
        them inside the predicate instead rescans every session event once per
        user response event, which is quadratic in the session size.
        """
        nonlocal branch_fc_ids
        if branch_fc_ids is None:
          descendant_prefix = f"{self.branch}."
          branch_fc_ids = {
              fc.id
              for branch_event in self.session.events
              if branch_event.branch
              and (
                  branch_event.branch == self.branch
                  or branch_event.branch.startswith(descendant_prefix)
              )
              for fc in branch_event.get_function_calls()
              if fc.id is not None
          }
        return branch_fc_ids

      def _is_branch_match(event: Event) -> bool:
        """Determines whether an event is part of this invocation's subtree.

        The rule differs by author, deliberately but asymmetrically.

        A user event matches when it sits on this branch, on a descendant
        sub-branch (e.g. a child NodeTool/WorkflowTool execution tree), or on
        no branch at all; and when ``self.branch`` is ``None``, every user
        event matches whatever branch it is on. An empty-string branch is the
        opposite rather than a synonym for that: it is a real branch value in
        the workflow code and matches no branched event. A user event carrying
        function responses must additionally answer a function call issued on
        this branch or below it -- one answering a call from anywhere else is
        dropped even when it sits on exactly this branch, which is what keeps
        a reply from leaking across parallel trees.

        Any other event must sit on exactly this branch; a descendant's own
        events are not returned.

        So a confirmation answered by the user on a sub-branch is visible here
        while the agent event that requested it is not. Widening the non-user
        rule to descendants would not expose sibling trees --
        ``_BranchPath.is_descendant_of`` excludes those -- but it would hand
        every caller a descendant's internal events, so it needs a per-caller
        review rather than a blanket change.

        Note this is the opposite direction from
        ``contents._is_event_belongs_to_branch``, which asks a different
        question -- "what history may this agent see?" -- and so matches
        *ancestor* branches instead. Both are intended.
        """
        if getattr(event, "author", None) == "user":
          frs = event.get_function_responses()
          if frs and self.branch and self.session:
            fr_ids = {fr.id for fr in frs if fr.id is not None}
            # If user's response IDs do not match any function call on this
            # branch tree, prevent event leakage across parallel or unrelated
            # branches.
            if fr_ids and not (fr_ids & _branch_function_call_ids()):
              return False

          # Match events yielded directly on this branch or on descendant
          # sub-branches (e.g. child NodeTool/WorkflowTool execution trees).
          # The `self.branch` guard keeps an empty branch from matching every
          # branched event, which a bare descendant test would do.
          if (
              event.branch is None
              or self.branch is None
              or event.branch == self.branch
              or (
                  self.branch
                  and _BranchPath.from_string(event.branch).is_descendant_of(
                      _BranchPath.from_string(self.branch)
                  )
              )
          ):
            return True
          return False
        # Non-user events: exactly this branch, per the docstring above.
        return event.branch == self.branch

      results = [e for e in results if _is_branch_match(e)]
    return results

  def should_pause_invocation(self, event: Event) -> bool:
    """Returns whether to pause the invocation right after this event.

    "Pausing" an invocation is different from "ending" an invocation. A paused
    invocation can be resumed later, while an ended invocation cannot.

    Pausing the current agent's run will also pause all the agents that
    depend on its execution, i.e. the subsequent agents in a workflow, and the
    current agent's ancestors, etc.

    Note that parallel sibling agents won't be affected, but their common
    ancestors will be paused after all the non-blocking sub-agents finished
    running.

    Should meet all following conditions to pause an invocation:
      1. The current event has a long running function call.

    Args:
      event: The current event.

    Returns:
      Whether to pause the invocation right after this event.
    """
    if not event.long_running_tool_ids or not event.get_function_calls():
      return False

    events = self.session.events if self.session else []
    for fc in event.get_function_calls():
      if fc.id in event.long_running_tool_ids:
        # Check if there is a newer user event in the session that belongs to a sub-branch of this tool call.
        # This indicates the tool call is resuming to process that nested input.
        is_resolving_sub_branch = False
        event_index = -1
        # Search backwards since the checked event is typically near the end of history.
        for i in range(len(events) - 1, -1, -1):
          if events[i].id == event.id:
            event_index = i
            break
        if event_index != -1:
          is_resolving_sub_branch = any(
              e.author == "user"
              and e.branch
              and fc.id in _BranchPath.from_string(e.branch).run_ids
              for e in events[event_index + 1 :]
          )

        if not is_resolving_sub_branch:
          return True

    return False

  # TODO: Move this method from invocation_context to a dedicated module.
  def _find_matching_function_call(
      self, function_response_event: Event
  ) -> Event | None:
    """Finds the function call event in the current invocation that matches the function response id."""
    from ..flows.llm_flows.functions import find_event_by_function_call_id

    function_responses = function_response_event.get_function_responses()
    if not function_responses:
      return None

    events = self._get_events(current_invocation=True)
    if events and events[-1].id == function_response_event.id:
      search_space = events[:-1]
    else:
      search_space = events

    function_response_id = function_responses[0].id
    if not function_response_id:
      return None
    return find_event_by_function_call_id(search_space, function_response_id)

  def stamp_event_branch_context(self, event: Event) -> None:
    """Stamps the event with the branch and isolation scope of its matching function call."""
    if function_call := self._find_matching_function_call(event):
      event.branch = function_call.branch
      if (
          event.isolation_scope is None
          and function_call.isolation_scope is not None
      ):
        event.isolation_scope = function_call.isolation_scope


def new_invocation_context_id() -> str:
  return "e-" + platform_uuid.new_uuid()
