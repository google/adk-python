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
from contextlib import aclosing
import json
from typing import Any
from typing import Optional

from google.genai import types
from pydantic import BaseModel

from ..agents.context import Context
from ..agents.llm.task._finish_task_tool import FINISH_TASK_SUCCESS_RESULT
from ..agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME as _FINISH_TASK_FC_NAME
from ..events.event import Event
from ..utils._schema_utils import validate_schema


def _extract_finish_task_fc(event: Event) -> Optional[types.FunctionCall]:
  """Returns the finish_task FC in this event, or None."""
  for fc in event.get_function_calls():
    if fc.name == _FINISH_TASK_FC_NAME:
      return fc
  return None


def _is_finish_task_success_fr(event: Event) -> bool:
  """True iff this event is the success FR from FinishTaskTool.

  A non-success FR (e.g., validation error) returns False so the
  caller keeps iterating and the LLM gets a chance to retry.
  """
  for fr in event.get_function_responses():
    if fr.name == _FINISH_TASK_FC_NAME:
      response = fr.response or {}
      return response.get('result') == FINISH_TASK_SUCCESS_RESULT
  return False


def _extract_task_delegation_fcs(
    event: Event, tools_dict: dict
) -> list[types.FunctionCall]:
  """Return task-delegation FCs from this event.

  A task-delegation FC is one whose tool is a ``_TaskAgentTool`` instance.
  """
  from ..tools.agent_tool import _TaskAgentTool

  return [
      fc
      for fc in event.get_function_calls()
      if fc.id
      and fc.name in tools_dict
      and isinstance(tools_dict[fc.name], _TaskAgentTool)
  ]


def _find_unresolved_task_delegations(
    session, owner: str, tools_dict: dict
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
      if (
          fc
          and fc.id
          and fc.name in tools_dict
          and isinstance(tools_dict[fc.name], _TaskAgentTool)
      ):
        fc_by_id[fc.id] = fc
      fr = part.function_response
      if fr and fr.id:
        fr_ids.add(fr.id)
  return [fc for fc_id, fc in fc_by_id.items() if fc_id not in fr_ids]


def _find_finish_task_tool(agent: Any) -> Any:
  """Return the FinishTaskTool instance attached to a task-mode agent."""
  for tool in getattr(agent, 'tools', []) or []:
    if getattr(tool, 'name', None) == _FINISH_TASK_FC_NAME:
      return tool
  return None


def _safe_canonical_tools_dict(agent: Any) -> dict:
  """Build a name→tool map from ``agent.tools``.

  Used by the chat wrapper to identify task-delegation FCs by tool
  name without resolving the agent's full canonical-tools pipeline.
  """
  out: dict = {}
  for tool in getattr(agent, 'tools', []) or []:
    name = getattr(tool, 'name', None)
    if name:
      out[name] = tool
  return out


async def _dispatch_task_fc(
    parent_agent: Any, fc: types.FunctionCall, ctx: Context
) -> Any:
  """Dispatch a task-delegation FC via ``ctx.run_node`` and return the output.

  ``run_id=fc.id`` makes the child run idempotent across resumes (same
  FC always maps to the same scheduler-tracked child run).  Scope is
  carried by ``isolation_scope`` (``override_isolation_scope=fc.id``); we
  intentionally do NOT set a branch — task-mode and single_turn-mode
  agents share the parent's branch and rely on isolation_scope for
  scoping instead.
  """
  target_agent = parent_agent.root_agent.find_agent(fc.name)
  if target_agent is None:
    raise ValueError(f'Task target agent {fc.name!r} not found.')
  from .utils._workflow_graph_utils import build_node

  wrapped_target = build_node(target_agent)
  wrapped_target.parent_agent = target_agent.parent_agent
  return await ctx.run_node(
      wrapped_target,
      node_input=fc.args,
      run_id=fc.id,
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


def _node_input_to_content(node_input: Any) -> types.Content:
  """Converts node_input to a user Content for the LLM agent."""
  if isinstance(node_input, types.Content):
    return types.Content(role='user', parts=node_input.parts)
  if isinstance(node_input, str):
    text = node_input
  elif isinstance(node_input, BaseModel):
    text = node_input.model_dump_json()
  elif isinstance(node_input, (dict, list)):
    text = json.dumps(node_input)
  else:
    text = str(node_input)
  return types.Content(role='user', parts=[types.Part(text=text)])


def prepare_llm_agent_context(agent: Any, ctx: Context) -> Context:
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

  ic.session = ic.session.model_copy(deep=False)
  return agent_ctx


def prepare_llm_agent_input(agent: Any, ctx: Context, node_input: Any) -> None:
  """Prepares the input for running LlmAgent as a node.

  For ``single_turn`` mode, append a user-role event with the input
  directly to session.events (legacy behavior).

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
  if node_input is None or agent.mode != 'single_turn':
    return
  agent_input = _node_input_to_content(node_input)
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


def process_llm_agent_output(agent: Any, ctx: Context, event: Event) -> None:
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
    agent: Any,
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
  update = {'agent': agent}
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
    update['user_content'] = _node_input_to_content(node_input)
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
          for fc in task_fcs:
            output = await _dispatch_task_fc(agent, fc, ctx)
            yield _synthesize_task_fr_event(fc, output)
          if task_fcs:
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
  pending_fc_args: Optional[dict] = None
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

      if pending_fc_args is not None and _is_finish_task_success_fr(event):
        wrapper_key = getattr(finish_tool, '_wrapper_key', None)
        if wrapper_key and wrapper_key in pending_fc_args:
          event.output = pending_fc_args[wrapper_key]
        else:
          event.output = pending_fc_args
        yield event
        return

      yield event
