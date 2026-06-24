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

"""Context class for ADK agents."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

from opentelemetry import context as context_api
from typing_extensions import override

from .readonly_context import ReadonlyContext

if TYPE_CHECKING:
  from google.genai import types

  from ..artifacts.base_artifact_service import ArtifactVersion
  from ..auth.auth_credential import AuthCredential
  from ..auth.auth_tool import AuthConfig
  from ..events.event import Event
  from ..events.event_actions import EventActions
  from ..events.ui_widget import UiWidget
  from ..memory.base_memory_service import SearchMemoryResponse
  from ..memory.memory_entry import MemoryEntry
  from ..sessions.session import Session
  from ..sessions.state import State
  from ..telemetry.node_tracing import TelemetryContext
  from ..tools.tool_confirmation import ToolConfirmation
  from ..workflow._base_node import BaseNode
  from ..workflow._graph import NodeLike
  from ..workflow._graph import RouteValue
  from ..workflow._schedule_dynamic_node import ScheduleDynamicNode
  from .invocation_context import InvocationContext

_MAX_PARENT_DEPTH = 50


def _derive_scheduler(
    parent_ctx: Context | None,
) -> ScheduleDynamicNode | None:
  """Derives the dynamic node scheduler from the parent context."""
  if parent_ctx:
    scheduler = parent_ctx._workflow_scheduler
    if scheduler is None:
      from ..workflow._dynamic_node_scheduler import DynamicNodeScheduler
      from ..workflow._dynamic_node_scheduler import DynamicNodeState

      scheduler = DynamicNodeScheduler(state=DynamicNodeState())
    return scheduler
  return None


def _derive_node_path(
    node_name: str | None,
    run_id: str,
    node_path: str | None,
    parent_path: str | None,
    *,
    node: BaseNode | None = None,
) -> tuple[str, str]:
  """Derives the node path and run ID."""
  if node_path:
    return node_path, run_id

  # Fallback: Reconstruct parent_path from static parent_agent Tree
  # if parent_path is missing during multi-turn session resumption.
  from ..agents.base_agent import BaseAgent
  from ..events._node_path_builder import _NodePathBuilder

  derived_run_id = run_id or '1'

  if not parent_path and isinstance(node, BaseAgent) and node.parent_agent:
    path_builder = _NodePathBuilder([])
    curr = node.parent_agent
    parent_agents = []
    depth = 0
    while curr is not None and depth < _MAX_PARENT_DEPTH:
      parent_agents.insert(0, curr)
      curr = curr.parent_agent
      depth += 1
    for agent in parent_agents:
      path_builder = path_builder.append(agent.name, '1')
    parent_path = str(path_builder)

  # Root contexts have no node name and no parent path. Return an empty path
  # to ensure they are correctly identified as the root of the execution
  # hierarchy.
  if not node_name and not parent_path:
    return '', derived_run_id

  base_path_builder = (
      _NodePathBuilder.from_string(parent_path)
      if parent_path
      else _NodePathBuilder([])
  )

  derived_node_path = str(
      base_path_builder.append(node_name or '', derived_run_id)
  )
  return derived_node_path, derived_run_id


class Context(ReadonlyContext):
  """The context within an agent run.

  When used in a workflow, additional fields under the ``Workflow-specific
  fields`` section are available.
  """

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      # Core State & Actions
      event_actions: EventActions | None = None,
      # Tool Execution
      function_call_id: str | None = None,
      tool_confirmation: ToolConfirmation | None = None,
      # Workflow Execution
      parent_ctx: Context | None = None,
      node: BaseNode | None = None,
      node_path: str | None = None,
      run_id: str = '',
      resume_inputs: dict[str, Any] | None = None,
      attempt_count: int = 1,
      use_as_output: bool = False,
  ) -> None:
    """Initializes the Context.

    Args:
      invocation_context: The invocation context.
      event_actions: The event actions for state and artifact deltas.
      function_call_id: The function call id of the current tool call. Required
        for tool-specific methods like request_credential and
        request_confirmation.
      tool_confirmation: The tool confirmation of the current tool call.
      parent_ctx: The parent node's Context.
      node: The current node.
      node_path: The path of the current node in the workflow graph. If not
        provided, it will be derived from parent_ctx and node.
      run_id: The execution ID of the current node.
      resume_inputs: Inputs for resuming node, keyed by interrupt id.
      attempt_count: Number of times this node has been attempted.
      use_as_output: If True, this node's output also represents the parent
        node's output.
    """
    super().__init__(invocation_context)

    self._parent_ctx = parent_ctx
    self._node = node

    from ..events.event_actions import EventActions
    from ..sessions.state import State
    from ..telemetry.node_tracing import TelemetryContext

    # Core State & Actions, Event & Telemetry
    self._event_actions = event_actions or EventActions()

    computed_state_schema = None
    if node and node.state_schema:
      computed_state_schema = node.state_schema
    elif parent_ctx:
      computed_state_schema = parent_ctx.state._schema

    self._state = State(
        value=invocation_context.session.state,
        delta=self._event_actions.state_delta,
        schema=computed_state_schema
        or getattr(invocation_context, '_state_schema', None),
    )

    self._event_author = parent_ctx.event_author if parent_ctx else ''

    self._telemetry_context = TelemetryContext(
        otel_context=context_api.get_current()
    )

    # Tool Execution
    self._function_call_id = function_call_id
    self._tool_confirmation = tool_confirmation

    # Workflow Execution
    self._node_path, self._run_id = _derive_node_path(
        node.name if node else None,
        run_id,
        node_path,
        parent_ctx.node_path if parent_ctx else None,
        node=node,
    )
    self._resume_inputs = resume_inputs or {}
    self._workflow_scheduler = _derive_scheduler(parent_ctx)
    self._node_rerun_on_resume = node.rerun_on_resume if node else True
    self._child_run_counters: dict[str, int] = {}
    self._attempt_count = attempt_count
    self._output_delegated = False
    self._output_value: Any = None
    self._output_emitted: bool = False
    self._route_value: RouteValue | list[RouteValue] | None = None
    self._route_emitted: bool = False
    self._interrupt_ids: set[str] = set()
    # scope tag inherited from parent ctx by default;
    # NodeRunner / Workflow may override before the node runs.
    self._isolation_scope: str | None = (
        parent_ctx.isolation_scope if parent_ctx else None
    )

    self._output_for_ancestors: list[str]
    if use_as_output and parent_ctx:
      self._output_for_ancestors = [parent_ctx.node_path] + list(
          parent_ctx._output_for_ancestors or []
      )
    else:
      self._output_for_ancestors = []
    self._error: Exception | None = None
    self._error_node_path: str = ''

  @property
  def function_call_id(self) -> str | None:
    """The function call id of the current tool call."""
    return self._function_call_id

  @function_call_id.setter
  def function_call_id(self, value: str | None) -> None:
    """Sets the function call id of the current tool call."""
    self._function_call_id = value

  @property
  def isolation_scope(self) -> str | None:
    """Scope tag inherited from parent or set explicitly via override.

    See ``Event.isolation_scope`` for format.

    ⚠️ DO NOT USE THIS DIRECTLY.  Internal mechanism, may change.
    """
    return self._isolation_scope

  @isolation_scope.setter
  def isolation_scope(self, value: str | None) -> None:
    self._isolation_scope = value

  @property
  def tool_confirmation(self) -> ToolConfirmation | None:
    """The tool confirmation of the current tool call."""
    return self._tool_confirmation

  @tool_confirmation.setter
  def tool_confirmation(self, value: ToolConfirmation | None) -> None:
    """Sets the tool confirmation of the current tool call."""
    self._tool_confirmation = value

  @property
  @override
  def state(self) -> State:
    """The delta-aware state of the current session.

    For any state change, you can mutate this object directly,
    e.g. `ctx.state['foo'] = 'bar'`
    """
    return self._state

  @property
  def actions(self) -> EventActions:
    """The event actions for the current context."""
    return self._event_actions

  @property
  @override
  def session(self) -> Session:
    """Returns the current session for this invocation."""
    return self._invocation_context.session

  # ============================================================================
  # Workflow-specific properties and methods
  # ============================================================================

  @property
  def parent_ctx(self) -> Context | None:
    """Returns the parent node's Context."""
    return self._parent_ctx

  @property
  def node(self) -> BaseNode | None:
    """Returns the node instance of this context."""
    return self._node

  @property
  def node_path(self) -> str:
    """Returns the path of the current node in the workflow graph."""
    return self._node_path

  @property
  def run_id(self) -> str:
    """Returns the execution ID of the current node."""
    return self._run_id

  @property
  def attempt_count(self) -> int:
    """Returns the current attempt number (1-based)."""
    return self._attempt_count

  @property
  def resume_inputs(self) -> dict[str, Any]:
    """Returns inputs for resuming node, keyed by interrupt id."""
    return self._resume_inputs

  @property
  def error(self) -> Exception | None:
    """The exception raised by the node, if any."""
    return self._error

  @property
  def error_node_path(self) -> str:
    """The path of the node that failed."""
    return self._error_node_path

  @property
  def output(self) -> Any:
    """The node's result value. Source of truth for node output.

    Set once per run. Also set by the framework when the node
    yields Event(output=X) or yields a raw value. If the value was
    set via yield, the output Event is already enqueued. If set
    directly, the framework emits the output Event after _run_impl
    returns.

    Raises ValueError if:
    - Set a second time (at most one output per execution).
    - Set when interrupt_ids is non-empty (output and interrupt
      are mutually exclusive).
    """
    return self._output_value

  @output.setter
  def output(self, value: Any) -> None:
    if self._output_value is not None:
      raise ValueError(
          'Output already set. A node can produce at most one output.'
      )
    self._output_value = value

  @property
  def route(self) -> RouteValue | list[RouteValue] | None:
    """Routing value for conditional edges.

    Read by the orchestrator to decide which downstream edge to
    follow. Can be set independently of output.
    """
    return self._route_value

  @route.setter
  def route(self, value: RouteValue | list[RouteValue]) -> None:
    self._route_value = value
    self._route_emitted = False

  @property
  def interrupt_ids(self) -> set[str]:
    """Interrupt IDs accumulated during this execution. Read-only.

    Set by the framework when the node yields an Event with
    long_running_tool_ids.
    """
    return set(self._interrupt_ids)

  @property
  def event_author(self) -> str:
    """Author name stamped on events emitted by this node.

    Set by the orchestrator to override the default (node name).
    For example, Workflow sets this to its own name so all child
    events appear under the workflow's author.

    Empty string means use the node's own name (default).
    """
    return self._event_author

  @event_author.setter
  def event_author(self, value: str) -> None:
    self._event_author = value

  @property
  def telemetry_context(self) -> TelemetryContext:
    """Returns the telemetry context."""
    return self._telemetry_context

  def get_invocation_context(self) -> InvocationContext:
    """Returns a copy of the invocation context with the proxy session."""
    ctx = self._invocation_context
    ctx_with_proxy = ctx.model_copy(
        update={
            'session': self.session,
            'isolation_scope': self.isolation_scope,
        }
    )
    return ctx_with_proxy

  async def run_node(
      self,
      node: NodeLike,
      node_input: Any = None,
      *,
      use_as_output: bool = False,
      run_id: str | None = None,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
      raise_on_wait: bool = False,
  ) -> Any:
    """Executes a node dynamically.

    This method allows a node within a workflow to trigger the run of
    another node (or a callable that can be built into a node) and
    asynchronously wait for its result. The dynamically executed node becomes
    a child run of the current node in the workflow.

    IMPORTANT: Always ``await`` this method directly. Wrapping it in
    ``asyncio.create_task()`` means the task runs unsupervised — errors
    are silently swallowed and the task is not cancelled if the parent
    node is interrupted (e.g. via HITL).

    Args:
      node: The node to be executed. This can be a BaseNode instance or a
        callable that can be built into a node.
      node_input: The input data to be passed to the dynamically executed node.
        Defaults to None.
      use_as_output: If True, the dynamic node's output is used as the
        calling node's output. The calling node's own output event is
        suppressed to avoid duplication.
      run_id: An optional custom run ID for the dynamic node execution.
        If not provided, a default run ID is generated. Useful for
        correlating events across runs.
      use_sub_branch: If True, the dynamic node will be executed in a sub-branch
        to isolate its state and events from the main branch.
      override_branch: An optional branch to use instead of parent's branch.

    Returns:
      The output of the dynamically executed node, once it finishes executing.
    """

    if not self._node_rerun_on_resume:
      raise ValueError(
          'A node must have rerun_on_resume=True. Reason is that dynamically'
          ' scheduled nodes might be interrupted, and the workflow'
          ' wakes-up/re-runs the parent node, so it can get the child node'
          ' response.'
      )

    from ..workflow.utils._workflow_graph_utils import build_node  # pylint: disable=g-import-not-at-top

    built_node = build_node(node)

    from ..agents.base_agent import BaseAgent

    if isinstance(node, BaseAgent) and isinstance(built_node, BaseAgent):
      built_node.parent_agent = node.parent_agent

    # Mode 1: Running within a Workflow graph.
    # The workflow orchestrator provides a scheduler to handle resume, dedup,
    # etc.
    if self._workflow_scheduler:
      from ..workflow._errors import NodeInterruptedError

      # Output delegation: once set, the calling node's own output
      # events are suppressed — the child's output (annotated with
      # output_for) becomes the calling node's output.
      if use_as_output:
        if self._output_delegated:
          raise ValueError(
              f'Node {self.node_path} already has a use_as_output delegate.'
          )
        self._output_delegated = True

      if run_id:
        if run_id.isdigit():
          raise ValueError(
              f'Explicit run_id "{run_id}" for node "{built_node.name}" must'
              ' contain non-numeric characters to prevent collision with'
              ' auto-generated IDs.'
          )
      else:
        self._child_run_counters[built_node.name] = (
            self._child_run_counters.get(built_node.name, 0) + 1
        )
        run_id = str(self._child_run_counters[built_node.name])

      child_ctx = await self._workflow_scheduler(
          self,
          built_node,
          node_input,
          node_name=built_node.name,
          use_as_output=use_as_output,
          run_id=run_id,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
      )
      if child_ctx.error:
        from ..workflow._errors import DynamicNodeFailError

        raise DynamicNodeFailError(
            message=f'Dynamic node {built_node.name} failed',
            error=child_ctx.error,
            error_node_path=child_ctx.error_node_path,
        )
      if child_ctx.interrupt_ids:
        # Propagate child's interrupt_ids to this node's ctx
        # so NodeRunner sees them after catching the error.
        self._interrupt_ids.update(child_ctx.interrupt_ids)
        raise NodeInterruptedError()
      # When the caller passes raise_on_wait=True, surface a child
      # that's WAITING (wait_for_output, no output, not transferring)
      # as NodeInterruptedError so the parent's NodeRunner records
      # the parent as WAITING instead of falsely COMPLETED.
      if (
          raise_on_wait
          and built_node.wait_for_output
          and child_ctx.output is None
          and not child_ctx.actions.transfer_to_agent
      ):
        raise NodeInterruptedError()
      return child_ctx.output

    # Mode 2: Standalone execution (outside of workflow).
    # Run the node directly via NodeRunner.
    result = await self._run_node_standalone(
        built_node,
        node_input,
        use_as_output=use_as_output,
        use_sub_branch=use_sub_branch,
        override_branch=override_branch,
        override_isolation_scope=override_isolation_scope,
        run_id=run_id,
    )
    if result.error:
      from ..workflow import _errors

      raise _errors.DynamicNodeFailError(
          message=f'Dynamic node {built_node.name} failed',
          error=result.error,
          error_node_path=result.error_node_path,
      )
    if (
        raise_on_wait
        and built_node.wait_for_output
        and result.output is None
        and not result.actions.transfer_to_agent
    ):
      from ..workflow._errors import NodeInterruptedError

      raise NodeInterruptedError()
    return result.output

  # ============================================================================
  # Artifact methods
  # ============================================================================

  async def load_artifact(
      self, filename: str, version: int | None = None
  ) -> types.Part | None:
    """Loads an artifact attached to the current session.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.load_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def save_artifact(
      self,
      filename: str,
      artifact: types.Part,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    """Saves an artifact and records it as delta for the current session.

    Args:
      filename: The filename of the artifact.
      artifact: The artifact to save.
      custom_metadata: Custom metadata to associate with the artifact.

    Returns:
     The version of the artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    version = await self._invocation_context.artifact_service.save_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        artifact=artifact,
        custom_metadata=custom_metadata,
    )
    self._event_actions.artifact_delta[filename] = version
    return version

  async def get_artifact_version(
      self, filename: str, version: int | None = None
  ) -> ArtifactVersion | None:
    """Gets artifact version info.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact version info.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.get_artifact_version(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def list_artifacts(self) -> list[str]:
    """Lists the filenames of the artifacts attached to the current session."""
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.list_artifact_keys(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
    )

  # ============================================================================
  # Credential methods
  # ============================================================================

  async def save_credential(self, auth_config: AuthConfig) -> None:
    """Saves a credential to the credential service.

    Args:
      auth_config: The authentication configuration containing the credential.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError('Credential service is not initialized.')
    await self._invocation_context.credential_service.save_credential(
        auth_config, self
    )

  async def load_credential(
      self, auth_config: AuthConfig
  ) -> AuthCredential | None:
    """Loads a credential from the credential service.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The loaded credential, or None if not found.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError('Credential service is not initialized.')
    return await self._invocation_context.credential_service.load_credential(
        auth_config, self
    )

  def get_auth_response(self, auth_config: AuthConfig) -> AuthCredential | None:
    """Gets the auth response credential from session state.

    This method retrieves an authentication credential that was previously
    stored in session state after a user completed an OAuth flow or other
    authentication process.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The auth credential from the auth response, or None if not found.
    """
    from ..auth.auth_handler import AuthHandler

    return AuthHandler(auth_config).get_auth_response(self.state)

  def request_credential(self, auth_config: AuthConfig) -> None:
    """Requests a credential for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set. For callback contexts, use save_credential/load_credential instead.

    Args:
      auth_config: The authentication configuration for the credential.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..auth.auth_handler import AuthHandler

    if not self.function_call_id:
      raise ValueError(
          'request_credential requires function_call_id. '
          'This method can only be used in a tool context, not a callback '
          'context. Consider using save_credential/load_credential instead.'
      )
    self._event_actions.requested_auth_configs[self.function_call_id] = (
        AuthHandler(auth_config).generate_auth_request()
    )

  # ============================================================================
  # Tool methods
  # ============================================================================

  def request_confirmation(
      self,
      *,
      hint: str | None = None,
      payload: Any | None = None,
  ) -> None:
    """Requests confirmation for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set.

    Args:
      hint: A hint to the user on how to confirm the tool call.
      payload: The payload used to confirm the tool call.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..tools.tool_confirmation import ToolConfirmation

    if not self.function_call_id:
      raise ValueError(
          'request_confirmation requires function_call_id. '
          'This method can only be used in a tool context.'
      )
    self._event_actions.requested_tool_confirmations[self.function_call_id] = (
        ToolConfirmation(
            hint=hint,
            payload=payload,
        )
    )

  # ============================================================================
  # Memory methods
  # ============================================================================

  async def add_session_to_memory(self) -> None:
    """Triggers memory generation for the current session.

    This method saves the current session's events to the memory service,
    enabling the agent to recall information from past interactions.

    Raises:
      ValueError: If memory service is not available.

    Example:
      ```python
      async def my_after_agent_callback(ctx: Context):
          # Save conversation to memory at the end of each interaction
          await ctx.add_session_to_memory()
      ```
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          'Cannot add session to memory: memory service is not available.'
      )
    await self._invocation_context.memory_service.add_session_to_memory(
        self._invocation_context.session
    )

  async def add_events_to_memory(
      self,
      *,
      events: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds an explicit list of events to the memory service.

    Uses this callback's current session identifiers as memory scope.

    Args:
      events: Explicit events to add to memory.
      custom_metadata: Optional metadata forwarded to the configured memory
        service. Supported keys are implementation-specific.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          'Cannot add events to memory: memory service is not available.'
      )
    await self._invocation_context.memory_service.add_events_to_memory(
        app_name=self._invocation_context.session.app_name,
        user_id=self._invocation_context.session.user_id,
        session_id=self._invocation_context.session.id,
        events=events,
        custom_metadata=custom_metadata,
    )

  async def add_memory(
      self,
      *,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds explicit memory items directly to the memory service.

    Uses this callback's current session identifiers as memory scope.

    Args:
      memories: Explicit memory items to add.
      custom_metadata: Optional metadata forwarded to the configured memory
        service. Supported keys are implementation-specific.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError('Cannot add memory: memory service is not available.')
    await self._invocation_context.memory_service.add_memory(
        app_name=self._invocation_context.session.app_name,
        user_id=self._invocation_context.session.user_id,
        memories=memories,
        custom_metadata=custom_metadata,
    )

  async def search_memory(self, query: str) -> SearchMemoryResponse:
    """Searches the memory of the current user.

    Args:
      query: The search query.

    Returns:
      The search results from the memory service.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError('Memory service is not available.')
    return await self._invocation_context.memory_service.search_memory(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        query=query,
    )

  # ============================================================================
  # UI Widget methods
  # ============================================================================

  def render_ui_widget(self, ui_widget: UiWidget) -> None:
    """Adds a UI widget to the current event's actions for the UI to render.

    UI widgets provide rendering payload/metadata that the UI Host uses to
    display rich interactive components (e.g., MCP App iframes) alongside agent
    responses.

    Args:
      ui_widget: A ``UiWidget`` instance.
    """
    if self._event_actions.render_ui_widgets is None:
      self._event_actions.render_ui_widgets = []

    for existing_widget in self._event_actions.render_ui_widgets:
      if existing_widget.id == ui_widget.id:
        raise ValueError(
            f"UI widget with ID '{ui_widget.id}' already exists in the current"
            ' event actions.'
        )

    self._event_actions.render_ui_widgets.append(ui_widget)

  # ============================================================================
  # Node Execution Dispatcher
  # ============================================================================

  async def _run_node_standalone(
      self,
      node: BaseNode,
      node_input: Any,
      *,
      use_as_output: bool = False,
      run_id: str | None = None,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
      resume_inputs: dict[str, Any] | None = None,
  ) -> Context:
    """Run a node directly via NodeRunner without an orchestrator."""
    from ..workflow._node_runner import NodeRunner

    runner = NodeRunner(
        node=node,
        parent_ctx=self,
        run_id=run_id,
        use_as_output=use_as_output,
        use_sub_branch=use_sub_branch,
        override_branch=override_branch,
        override_isolation_scope=override_isolation_scope,
    )
    return await runner.run(node_input=node_input, resume_inputs=resume_inputs)
