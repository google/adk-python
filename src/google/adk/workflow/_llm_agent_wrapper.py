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

"""Utility functions for running LlmAgent as a workflow node."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Mapping
from contextlib import aclosing
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from google.genai import types

from ..agents.context import Context
from ..agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME as _FINISH_TASK_FC_NAME
from ..agents.llm.task._finish_task_tool import is_finish_task_terminal_fr
from ..events.event import Event
from ..flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from ..utils._schema_utils import validate_schema
from ..utils.content_utils import to_user_content

if TYPE_CHECKING:
  from ..agents.llm_agent import LlmAgent
  from ..agents.llm_agent import ToolUnion
  from ..sessions.session import Session


def _extract_finish_task_fc(event: Event) -> types.FunctionCall | None:
  """Returns the finish_task FC in this event, or None.

  A partial event is skipped: its arguments may still be arriving, and
  promoting them would make a fragment decide the task's output.
  """
  if event.partial:
    return None
  for fc in event.get_function_calls():
    if fc.name == _FINISH_TASK_FC_NAME:
      return fc
  return None


def _extract_task_delegation_fcs(
    event: Event, tools_dict: Mapping[str, ToolUnion]
) -> list[types.FunctionCall]:
  """Return task-delegation FCs from this event.

  A task-delegation FC is one whose tool is a ``_TaskAgentTool`` instance.

  A partial event yields nothing, the way ``process_llm_agent_output``
  already treats one. Its arguments may still be arriving, and the Runner
  never appends a fragment to the session, so dispatching from one would run
  the specialist on truncated arguments and leave the delegation out of the
  history the coordinator replays.
  """
  if event.partial:
    return []

  from ..tools.agent_tool import _TaskAgentTool

  return [
      fc
      for fc in event.get_function_calls()
      if fc.id
      and fc.name is not None
      and isinstance(tools_dict.get(fc.name), _TaskAgentTool)
  ]


def _event_has_eager_tool_calls(
    event: Event, tools_dict: Mapping[str, ToolUnion]
) -> bool:
  """True if this event has FCs that produce FR events in the current step.

  Task-delegation tools (``_TaskAgentTool``) and other deferred / long-running
  tools do not emit an FR from ``handle_function_calls_async``; the chat
  wrapper synthesizes task FRs itself. Regular tools (including long-running or
  deferred tools that return a value) do emit FRs in the same LLM step, after
  the model FC event. The wrapper must drain those FR events before closing the
  generator, or they are lost and the session history becomes unbalanced for
  Gemini.

  Args:
    event: The event containing function calls.
    tools_dict: Map of tool names to Tool objects.

  Returns:
    True if the event has eager tool calls.
  """
  from ..tools.agent_tool import _TaskAgentTool  # pylint: disable=g-import-not-at-top

  for fc in event.get_function_calls():
    if not fc.name:
      continue
    tool = tools_dict.get(fc.name)
    if tool is None or isinstance(tool, _TaskAgentTool):
      continue
    return True
  return False


async def _drain_pending_tool_response_events(
    run_iter: AsyncGenerator[Event, None],
) -> AsyncGenerator[Event, None]:
  """Yield remaining non-model events from the current LLM step.

  After a mixed model turn (regular tools + task delegation), the LLM flow
  still has pending function-response events. Closing the generator before
  reading them drops regular-tool FRs.

  Stops after the first event that carries function responses, or before the
  next model-role event (which would start another LLM round without
  synthesized task FRs).

  Args:
    run_iter: The generator to drain events from.

  Yields:
    Events from the current LLM step.
  """
  async for pending_event in run_iter:
    if (
        pending_event.content is not None
        and pending_event.content.role == 'model'
    ):
      # Tool confirmation events have role 'model' but they are part of the
      # current step (asking for confirmation before executing the tool).
      # We must yield them and continue draining the actual FR.
      is_confirmation = any(
          fc.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
          for fc in pending_event.get_function_calls()
      )
      if is_confirmation:
        yield pending_event
        continue

      # Next LLM round already started; abandon it by stopping iteration.
      # Closing the outer generator cancels further work.
      return
    yield pending_event
    if pending_event.get_function_responses():
      return


def _find_unresolved_task_delegations(
    session: Session,
    owner: str,
    tools_dict: Mapping[str, ToolUnion],
) -> list[types.FunctionCall]:
  """Walk session events; find task FCs from ``owner`` without matching FRs.

  Sequential dispatch means at most one
  unresolved task delegation at a time, but we return a list so the
  caller can iterate uniformly.

  We deliberately do NOT filter by isolation_scope. A chat
  coordinator's conversation persists across user turns; each turn
  produces a fresh ``wf:<user_event_id>`` scope, so filtering by the
  current turn's scope would hide the coordinator's own FC from a
  prior turn.  Author + tool-name filtering is sufficient.
  """
  from ..tools.agent_tool import _TaskAgentTool

  fc_by_id: dict[str, types.FunctionCall] = {}
  fr_ids: set[str] = set()
  for event in session.events:
    if event.author != owner and event.author != 'user':
      continue
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      fc = part.function_call
      tool_name = fc.name if fc is not None else None
      if (
          fc
          and fc.id
          and tool_name is not None
          and isinstance(tools_dict.get(tool_name), _TaskAgentTool)
      ):
        fc_by_id[fc.id] = fc
      fr = part.function_response
      if fr and fr.id:
        fr_ids.add(fr.id)
  return [fc for fc_id, fc in fc_by_id.items() if fc_id not in fr_ids]


def _agent_tools(agent: LlmAgent) -> list[ToolUnion]:
  """Returns ``agent.tools``, tolerating agents that do not define it."""
  tools: list[ToolUnion] = getattr(agent, 'tools', None) or []
  return tools


def _find_finish_task_tool(agent: LlmAgent) -> ToolUnion | None:
  """Return the FinishTaskTool instance attached to a task-mode agent."""
  for tool in _agent_tools(agent):
    if getattr(tool, 'name', None) == _FINISH_TASK_FC_NAME:
      return tool
  return None


def _safe_canonical_tools_dict(agent: LlmAgent) -> dict[str, ToolUnion]:
  """Build a name→tool map from ``agent.tools``.

  Used by the chat wrapper to identify task-delegation FCs by tool
  name without resolving the agent's full canonical-tools pipeline.
  """
  out: dict[str, ToolUnion] = {}
  for tool in _agent_tools(agent):
    name = getattr(tool, 'name', None)
    if isinstance(name, str) and name:
      out[name] = tool
  return out


async def _dispatch_task_fc(
    parent_agent: LlmAgent, fc: types.FunctionCall, ctx: Context
) -> Any:
  """Dispatch a task-delegation FC via ``ctx.run_node`` and return the output.

  ``run_id=fc.id`` makes the child run idempotent across resumes (same
  FC always maps to the same scheduler-tracked child run).  Each task
  runs in a stable sub-branch so resumable LLM flow logic sees only the
  task's own function calls.  ``isolation_scope`` remains keyed by the
  FC id to keep task history scoped independently of branch ancestry.
  """
  if fc.name is None or fc.id is None:
    raise ValueError('Task delegation calls require both a name and an ID.')
  target_agent = parent_agent.root_agent.find_agent(fc.name)
  if target_agent is None:
    raise ValueError(f'Task target agent {fc.name!r} not found.')
  from .utils._workflow_graph_utils import build_node

  wrapped_target = build_node(target_agent)
  cast(Any, wrapped_target).parent_agent = target_agent.parent_agent
  return await ctx.run_node(
      wrapped_target,
      node_input=fc.args,
      run_id=fc.id,
      use_sub_branch=True,
      override_isolation_scope=fc.id,
      raise_on_wait=True,
  )


def _synthesize_task_fr_event(fc: types.FunctionCall, output: Any) -> Event:
  """Build the synthesized FR event for a completed task delegation.

  No isolation_scope is set on the event itself — ``NodeRunner._enrich_event``
  stamps it from the parent's ``ctx.isolation_scope`` (which is the
  coordinator's scope or None for a root chat coordinator).  This keeps
  the FR visible to the parent and invisible to other task scopes.
  """
  if isinstance(output, dict):
    response = output
  else:
    response = {'output': output}
  fr_part = types.Part(
      function_response=types.FunctionResponse(
          id=fc.id,
          name=fc.name,
          response=response,
      )
  )
  return Event(
      author='user',
      content=types.Content(role='user', parts=[fr_part]),
  )


def prepare_llm_agent_context(agent: LlmAgent, ctx: Context) -> Context:
  """Prepares the context for running LlmAgent as a node."""
  if agent.mode != 'single_turn':
    return ctx

  ic = ctx._invocation_context.model_copy()
  ic._event_queue = ctx._invocation_context._event_queue
  ic.isolation_scope = ctx.isolation_scope
  agent_ctx = Context(
      invocation_context=ic,
      node_path=ctx.node_path,
      run_id=ctx.run_id,
      resume_inputs=ctx.resume_inputs,
  )
  agent_ctx.isolation_scope = ctx.isolation_scope

  # Share the parent's `session` object (don't copy it): a mid-invocation
  # write such as compaction must be visible to later nodes, or the DB
  # service rejects their write as stale.
  return agent_ctx


def prepare_llm_agent_input(
    agent: LlmAgent, ctx: Context, node_input: object
) -> None:
  """Prepares the input for running LlmAgent as a node.

  For ``single_turn`` mode, append a user-role event with the input
  directly to session.events (legacy behavior). When resuming with
  ``resume_inputs``, skip appending to avoid injecting duplicate synthetic
  user events that shadow user function responses.

  For ``task`` mode, the input is the parent's task-delegation FC
  args.  Those are NOT appended here — the content-builder
  transforms the originating FC event into a leading user-role
  content at LLM-request time, so it appears as the first turn in
  the task agent's view.  When no originating FC exists (task agent
  dispatched directly as a Workflow node), the wrapper instead
  overrides ``ic.user_content`` so the content-builder can fall back
  to that as the first user turn.

  For workflow nodes running in a sub-branch, stamp the input event with that
  branch. A private node input should not look like the shared root user turn.
  """
  # Skip injection if:
  # 1. No input was provided.
  # 2. Agent is not single_turn (task mode handles its own leading turn).
  # 3. Resuming from pause/interrupt (e.g. HITL confirmation): node is re-run
  #    with resume_inputs; injecting synthetic user input would shadow the
  #    user's FunctionResponse on the branch tail and cause infinite loops.
  if (
      node_input is None
      or agent.mode != 'single_turn'
      or bool(ctx.resume_inputs)
  ):
    return
  agent_input = to_user_content(node_input)
  user_event = Event(author='user', message=agent_input)
  if user_event.content is not None:
    user_event.content.role = 'user'
  iso = getattr(ctx, 'isolation_scope', None)
  if iso:
    user_event.isolation_scope = iso
  branch = ctx._invocation_context.branch
  if branch:
    user_event.branch = branch
  ctx.session.events.append(user_event)


def process_llm_agent_output(
    agent: LlmAgent, ctx: Context, event: Event
) -> None:
  """Processes the output of LlmAgent run as a node."""
  if (
      event.get_function_calls()
      or event.partial
      or not event.content
      or event.content.role != 'model'
  ):
    return

  output = None
  text = (
      ''.join(p.text for p in event.content.parts if p.text and not p.thought)
      if event.content.parts
      else ''
  )
  if agent.output_schema:
    if text.strip():
      output = validate_schema(agent.output_schema, text)
    else:
      output = None
  else:
    output = text

  if agent.output_key and output is not None:
    ctx.actions.state_delta[agent.output_key] = output

  event.output = output
  event.node_info.message_as_output = True


async def run_llm_agent_as_node(
    agent: LlmAgent,
    *,
    ctx: Context,
    node_input: Any,
) -> AsyncGenerator[Any, None]:
  """Runs an LlmAgent as a workflow node."""
  # As a node in a workflow, agent is by default single_turn.
  if agent.mode is None:
    agent.mode = 'single_turn'

  if agent.mode not in ('task', 'single_turn', 'chat'):
    raise ValueError(
        f'LlmAgent as node only supports task, single_turn, and chat mode,'
        f" but agent '{agent.name}' has mode='{agent.mode}'."
    )

  include_contents_explicit = 'include_contents' in agent.model_fields_set
  if agent.mode == 'single_turn' and not include_contents_explicit:
    agent.include_contents = 'none'

  agent_ctx = prepare_llm_agent_context(agent, ctx)
  prepare_llm_agent_input(agent, agent_ctx, node_input)

  ic = agent_ctx.get_invocation_context()
  update: dict[str, object] = {'agent': agent}
  # thread the agent's isolation_scope into the
  # InvocationContext so the content processor can filter session
  # events to this agent's scope only.  Only mode=task and
  # mode=single_turn agents need scope-based filtering — chat agents
  # see the full conversation.
  _agent_iso = getattr(agent_ctx, 'isolation_scope', None)
  if agent.mode in ('task', 'single_turn') and _agent_iso:
    update['isolation_scope'] = _agent_iso
  # Override ``user_content`` for task mode with this node's input.
  # The content-builder uses it as the fallback first user turn when
  # there is no originating delegation FC (the workflow-node task
  # case).  For delegated tasks, the FC takes precedence and this
  # override is unused.
  if agent.mode == 'task' and node_input is not None:
    update['user_content'] = to_user_content(node_input)
  ic = ic.model_copy(update=update)

  from ..agents.live_request_queue import LiveRequestQueue

  # A single_turn LlmAgent in a live session runs in non-live mode
  # and only consumes the node_input (ignoring the live request queue).
  is_live = (
      isinstance(getattr(ic, 'live_request_queue', None), LiveRequestQueue)
      and agent.mode != 'single_turn'
  )

  if agent.mode == 'single_turn':
    # is_live is always False here (single_turn forces non-live).
    async with aclosing(agent.run_async(ic)) as run_iter:
      async for event in run_iter:
        process_llm_agent_output(agent, ctx, event)
        yield event
    return

  if agent.mode == 'chat':
    # outer dispatch loop.
    #
    # One coordinator invocation may contain multiple LLM rounds chained
    # by task delegations.  Example for sequential delegation:
    #
    #   1. (Optional) Pre-LLM scan: replay any unresolved task FCs from
    #      prior turns.  Their dispatched sub-agents may complete or
    #      raise NodeInterruptedError (still WAITING).
    #   2. Run parent.run_async: LLM emits a fresh task FC -> dispatch
    #      and synthesize FR.
    #   3. Re-enter parent.run_async with the FR now in session: LLM
    #      may emit another task FC, a normal tool, or natural text.
    #
    # The previous implementation broke after the first dispatch in
    # step 2, which prevented chained delegations from continuing
    # within the same invocation.  The outer ``while True`` loop fixes
    # that by re-entering ``agent.run_async`` after every task FC
    # dispatch, until the LLM returns without one.
    tools_dict = _safe_canonical_tools_dict(agent)

    # Step 1 (only on the very first iteration of this invocation):
    # pre-LLM scan for unresolved task FCs from prior runs.
    pending = _find_unresolved_task_delegations(
        ctx.session,
        owner=agent.name,
        tools_dict=tools_dict,
    )
    for fc in pending:
      output = await _dispatch_task_fc(agent, fc, ctx)
      yield _synthesize_task_fr_event(fc, output)

    # Step 2: run parent.run_async; on every fresh task FC, dispatch
    # and re-enter parent.run_async with the FR in session.
    while True:
      had_task_fc = False
      transferred = False
      run_method = agent.run_live(ic) if is_live else agent.run_async(ic)
      async with aclosing(run_method) as run_iter:
        async for event in run_iter:
          yield event
          task_fcs = _extract_task_delegation_fcs(event, tools_dict)
          if task_fcs:
            # Mixed turns (regular tool FC + task FC) still have pending
            # regular-tool FR events in this generator. Drain them before
            # breaking, otherwise aclosing drops them and the session is
            # left with unbalanced FC/FR history that Gemini rejects.
            if _event_has_eager_tool_calls(event, tools_dict):
              async with aclosing(
                  _drain_pending_tool_response_events(run_iter)
              ) as drain_iter:
                async for pending_event in drain_iter:
                  yield pending_event

            for fc in task_fcs:
              output = await _dispatch_task_fc(agent, fc, ctx)
              yield _synthesize_task_fr_event(fc, output)
            had_task_fc = True
            break  # close this run_iter; outer loop re-enters
          if event.actions.transfer_to_agent:
            target_name = event.actions.transfer_to_agent

            from ..agents.llm_agent import LlmAgent

            if (
                isinstance(agent, LlmAgent)
                and ctx._invocation_context.is_resumable
            ):
              ctx._invocation_context.set_agent_state(
                  agent.name, end_of_agent=True
              )
              yield agent._create_agent_state_event(ctx._invocation_context)
            transferred = True
            break
      if not had_task_fc or transferred:
        # LLM finished without delegating (or transferred away);
        # nothing more for this wrapper to do.
        return
      # Otherwise: loop back to re-enter agent.run_async so the LLM
      # sees the synthesized FR(s) and can emit follow-up actions.

  # Task mode: sniff the finish_task FC, but wait for FinishTaskTool's
  # FR before terminating.  If validation fails (FR carries an
  # ``error`` key), let the LLM see the error and retry on the next
  # round.  Only on a successful FR do we promote the FC's args as the
  # task output and exit.
  #
  # The finish_task tool's declaration mirrors the agent's
  # output_schema. For wrapped primitives (`int`, `str`, etc.) the
  # value lives at the wrapper key; for object schemas it's at the
  # top level of args. We extract via the FinishTaskTool's
  # `_wrapper_key` when accessible, falling back to the full args.
  finish_tool = _find_finish_task_tool(agent)
  pending_fc_args: dict[str, Any] | None = None
  run_method = agent.run_live(ic) if is_live else agent.run_async(ic)
  async with aclosing(run_method) as run_iter:
    async for event in run_iter:
      finish_fc = _extract_finish_task_fc(event)
      if finish_fc is not None:
        # Remember the latest FC's args; wait for FinishTaskTool's FR
        # before terminating.  If validation fails, the FR will NOT be
        # the success message — the LLM sees the error and retries.
        pending_fc_args = dict(finish_fc.args or {})
        yield event
        continue

      if pending_fc_args is not None and is_finish_task_terminal_fr(event):
        wrapper_key = getattr(finish_tool, '_wrapper_key', None)
        if wrapper_key and wrapper_key in pending_fc_args:
          event.output = pending_fc_args[wrapper_key]
        else:
          event.output = pending_fc_args
        output_key = getattr(agent, 'output_key', None)
        if output_key and event.output is not None:
          ctx.actions.state_delta[output_key] = event.output
        yield event
        return

      yield event
