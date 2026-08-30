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

"""Private helper module for workflow and node execution in ADK."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry import context

from ..agents.base_agent import BaseAgent
from ..agents.context import Context
from ..agents.run_config import RunConfig
from ..events.event import Event
from ..sessions.session import Session
from ..telemetry import _instrumentation
from ..utils._runner_utils import _notify_run_error
from ..utils._runner_utils import _with_caller_context
from ._dynamic_node_scheduler import DynamicNodeScheduler
from ._errors import DynamicNodeFailError
from ._errors import NodeInterruptedError
from ._workflow import _LoopState

if TYPE_CHECKING:
  from ..runners import Runner
  from ._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)


async def run_node_async(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    invocation_id: Optional[str] = None,
    new_message: Optional[types.Content] = None,
    state_delta: Optional[dict[str, Any]] = None,
    run_config: Optional[RunConfig] = None,
    yield_user_message: bool = False,
    node: BaseNode | None = None,
    session: Optional[Session] = None,
) -> AsyncGenerator[Event, None]:
  """Runs a BaseNode or Workflow in async mode."""
  from ..runners import _apply_run_config_custom_metadata
  from ..runners import _find_active_task_scope

  caller_ctx = context.get_current()

  async def _run() -> AsyncGenerator[Event, None]:
    nonlocal invocation_id, new_message, session
    with _instrumentation.record_invocation(
        entrypoint_node=node or runner.agent,
        conversation_id=session_id,
        run_config=run_config or RunConfig(),
    ):
      # 1. Setup
      if session is None:
        session = await runner._get_or_create_session(  # pylint: disable=protected-access
            user_id=user_id,
            session_id=session_id,
            get_session_config=(run_config or RunConfig()).get_session_config,
        )

      # Validate and resolve resume inputs
      resume_inputs = runner._extract_resume_inputs(new_message)  # pylint: disable=protected-access
      runner._validate_new_message(new_message, resume_inputs)  # pylint: disable=protected-access

      if not invocation_id and new_message:
        invocation_id = runner._resolve_invocation_id_from_fr(  # pylint: disable=protected-access
            session, new_message
        )
        if not invocation_id:
          active_scope = _find_active_task_scope(session)
          if active_scope:
            _, inv_id = active_scope
            invocation_id = inv_id
      elif invocation_id and new_message:
        # A caller-supplied id is reconciled against the responses rather
        # than trusted: resuming under an id that does not own the call
        # means the call is not found and the response is dropped, losing
        # the tool result. This is the same reconciliation the non-node
        # path performs, through the same helper, so a root LlmAgent gets
        # one answer no matter which path the runner picked for it.
        invocation_id = runner._resolve_invocation_id(  # pylint: disable=protected-access
            session, new_message, invocation_id
        )

      ic = runner._new_invocation_context(  # pylint: disable=protected-access
          session,
          new_message=new_message,
          run_config=run_config or RunConfig(),
          invocation_id=invocation_id,
      )
      if node and node is not runner.agent:
        ic.agent = node
        runner._restore_branch_from_history(  # pylint: disable=protected-access
            ic, node, root=runner.agent, invocation_id=invocation_id
        )
      ic._event_queue = asyncio.Queue()  # pylint: disable=protected-access

      # 2. Append user message to session and resolve node_input
      node_input = None
      if resume_inputs or invocation_id:
        node_input = runner._find_user_message_for_invocation(  # pylint: disable=protected-access
            ic.session.events, ic.invocation_id
        )
        if node_input:
          ic.user_content = node_input
      if not node_input:
        node_input = new_message

      # Failures in the setup hooks below (on_user_message_callback, the
      # user-event session append, and before_run_callback) must also notify
      # on_run_error_callback: they are part of runner execution even though
      # they run before the main event loop. Notification-only; the original
      # exception is always re-raised, and after_run stays success-only.
      run_error = None
      try:
        try:
          # Run callbacks on user message
          if new_message:
            modified_user_message = (
                await ic.plugin_manager.run_on_user_message_callback(
                    invocation_context=ic, user_message=new_message
                )
            )
            if modified_user_message is not None:
              new_message = modified_user_message
              ic.user_content = new_message

          # Append user message to session for history
          if new_message:
            user_event = await runner._append_user_event(  # pylint: disable=protected-access
                ic, new_message, state_delta=state_delta
            )
            if yield_user_message and user_event:
              yield user_event

          # Run before_run callbacks. A returned Content halts execution and ends
          # the run with that content (same contract as the non-workflow path).
          early_exit_result = await ic.plugin_manager.run_before_run_callback(
              invocation_context=ic
          )
          if isinstance(early_exit_result, types.Content):
            early_exit_event = Event(
                invocation_id=ic.invocation_id,
                author="model",
                content=early_exit_result,
            )
            _apply_run_config_custom_metadata(early_exit_event, ic.run_config)
            if runner._should_append_event(  # pylint: disable=protected-access
                early_exit_event, is_live_call=False
            ):
              await runner.session_service.append_event(
                  session=ic.session,
                  event=early_exit_event,
              )
            yield early_exit_event
          else:
            # 3. Start root node in background
            root_ctx = Context(ic)
            root_node = node or runner.agent
            is_agent = isinstance(runner.agent, BaseAgent)
            has_sub_agents = is_agent and bool(
                getattr(runner.agent, "sub_agents", None)
            )
            use_scheduler = is_agent and has_sub_agents

            # The root chat coordinator's isolation_scope stays None: its own
            # events (FCs, text, synthesized FRs from completed task
            # delegations) are also unscoped, so the content-builder's
            # isolation_scope filter lets the coordinator see all of them
            # across user turns. Task sub-agents are scoped under their
            # originating function-call id and so remain invisible to the
            # coordinator's view.

            done_sentinel = object()

            async def _drive_root_node() -> None:
              try:
                if use_scheduler:
                  # Rehydration warning: DynamicNodeScheduler relies on session.events scanning.
                  # Stateful live EUC/LRO streams may rehydrate freshly if not yet persisted.
                  scheduler = DynamicNodeScheduler(state=_LoopState())
                  root_ctx._workflow_scheduler = scheduler  # pylint: disable=protected-access

                try:
                  await root_ctx._run_node_internal(  # pylint: disable=protected-access
                      root_node,
                      node_input=node_input,
                      resume_inputs=resume_inputs,
                  )
                except NodeInterruptedError:
                  # The node was interrupted (e.g. for HITL).
                  pass
                except DynamicNodeFailError as e:
                  raise e.error
              finally:
                assert ic._event_queue is not None  # pylint: disable=protected-access
                await ic._event_queue.put((done_sentinel, None))  # pylint: disable=protected-access

            task = asyncio.create_task(_drive_root_node())

            # 4. Main loop: consume events, persist, yield
            try:
              async with aclosing(
                  runner._consume_event_queue(ic, done_sentinel)  # pylint: disable=protected-access
              ) as agen:
                async for event in agen:
                  yield event
            finally:
              # _cleanup_root_task re-raises a root-node Exception (if any) after
              # the event stream has drained.
              await runner._cleanup_root_task(task, runner.agent.name)  # pylint: disable=protected-access
        except Exception as e:
          # An unhandled exception escaped runner execution. Notify plugins
          # (notification-only) and re-raise. after_run stays success-only.
          run_error = e
          await _notify_run_error(ic.plugin_manager, ic, e)
          raise
      finally:
        # Success path (also caller early-stop via GeneratorExit, which is not
        # an Exception): run after_run and compaction. _cleanup_root_task has
        # already run in the inner finally above when a root task was created.
        # A failure in this success cleanup (e.g. an after_run plugin raising,
        # which PluginManager surfaces as a RuntimeError) is itself an
        # unhandled runner error, so notify on_run_error_callback once and
        # re-raise. on_run_error is notification-only and never raises, so
        # there is no recursive notification.
        if run_error is None:
          try:
            await ic.plugin_manager.run_after_run_callback(
                invocation_context=ic
            )
            await runner._run_post_invocation_compaction(  # pylint: disable=protected-access
                session=session,
                skip_token_compaction=ic.token_compaction_checked,
            )
          except Exception as e:
            await _notify_run_error(ic.plugin_manager, ic, e)
            raise

  async with aclosing(_with_caller_context(_run(), caller_ctx)) as agen:
    async for event in agen:
      yield event
