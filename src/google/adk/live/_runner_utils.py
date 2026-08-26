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

"""Live mode runner execution helpers extracted from google.adk.runners."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from opentelemetry import context

from ..agents.run_config import RunConfig
from ..events.event import Event
from ..utils._runner_utils import _notify_run_error
from ..utils._runner_utils import _with_caller_context
from .live_request_queue import LiveRequestQueue

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..runners import Runner
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


def new_invocation_context_for_live(
    runner: Runner,
    session: Session,
    *,
    live_request_queue: LiveRequestQueue,
    run_config: Optional[RunConfig] = None,
) -> InvocationContext:
  """Creates a new invocation context for live multi-agent."""
  run_config = run_config or RunConfig()

  # For live multi-agents system, we need model's text transcription as
  # context for the transferred agent.
  if hasattr(runner.agent, 'sub_agents') and runner.agent.sub_agents:
    if (
        run_config.response_modalities
        and types.Modality.AUDIO in run_config.response_modalities
    ):
      if not run_config.output_audio_transcription:
        run_config.output_audio_transcription = types.AudioTranscriptionConfig()
    if not run_config.input_audio_transcription:
      run_config.input_audio_transcription = types.AudioTranscriptionConfig()
  return runner._new_invocation_context(
      session,
      live_request_queue=live_request_queue,
      run_config=run_config,
  )


async def run_node_live(
    runner: Runner,
    *,
    session: Session,
    live_request_queue: LiveRequestQueue,
    run_config: Optional[RunConfig] = None,
) -> AsyncGenerator[Event, None]:
  """Runs a non-agent BaseNode in live mode."""
  from ..agents.context import Context
  from ..workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from ..workflow._errors import DynamicNodeFailError
  from ..workflow._errors import NodeInterruptedError
  from ..workflow._workflow import _LoopState
  from ..workflow._workflow import Workflow

  ic = new_invocation_context_for_live(
      runner,
      session,
      live_request_queue=live_request_queue,
      run_config=run_config or RunConfig(),
  )
  ic._event_queue = asyncio.Queue()

  root_ctx = Context(ic)
  root_agent = runner.agent
  is_workflow = isinstance(root_agent, Workflow)

  done_sentinel = object()

  async def _drive_root_node() -> None:
    try:
      if is_workflow:
        scheduler = DynamicNodeScheduler(state=_LoopState())
        root_ctx._workflow_scheduler = scheduler

      try:
        await root_ctx.run_node(
            root_agent,
            node_input=None,
        )
      except NodeInterruptedError:
        pass
      except DynamicNodeFailError as e:
        raise e.error
    finally:
      assert ic._event_queue is not None
      await ic._event_queue.put((done_sentinel, None))

  task = asyncio.create_task(_drive_root_node())

  try:
    try:
      async with aclosing(
          runner._consume_event_queue(ic, done_sentinel)
      ) as agen:
        async for event in agen:
          yield event
    finally:
      # _cleanup_root_task re-raises a root-node Exception (if any).
      await runner._cleanup_root_task(task, runner.agent.name)
  except Exception as e:
    # An unhandled exception escaped live runner execution. Notify plugins
    # (notification-only) and re-raise.
    await _notify_run_error(ic.plugin_manager, ic, e)
    raise


async def run_live(
    runner: Runner,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    live_request_queue: Optional[LiveRequestQueue] = None,
    run_config: Optional[RunConfig] = None,
    session: Optional[Session] = None,
) -> AsyncGenerator[Event, None]:
  """Runs the runner's agent in live (bidirectional streaming) mode."""
  run_config = run_config or RunConfig()
  # Some native audio models requires the modality to be set. So we set it to
  # AUDIO by default.
  #
  # The default goes on a copy rather than on the caller's own RunConfig: a
  # config that asked for nothing in particular would otherwise come back out
  # of the run pinned to AUDIO, and a config reused for a later run would
  # carry that choice into it. The copy is shallow on purpose. Deep copying a
  # RunConfig raises `TypeError: cannot pickle` when `http_options` holds a
  # live httpx client, and nothing here writes through into a sub-model.
  if run_config.response_modalities is None:
    run_config = run_config.model_copy()
    run_config.response_modalities = [types.Modality.AUDIO]

  caller_ctx = context.get_current()
  if session is None and (user_id is None or session_id is None):
    raise ValueError(
        'Either session or user_id and session_id must be provided.'
    )
  if live_request_queue is None:
    raise ValueError('live_request_queue is required for run_live.')
  if session is not None:
    warnings.warn(
        'The `session` parameter is deprecated. Please use `user_id` and'
        ' `session_id` instead.',
        DeprecationWarning,
        stacklevel=2,
    )
  if session is None:
    if user_id is None or session_id is None:
      raise ValueError(
          'user_id and session_id are required when session is not provided.'
      )
    session = await runner._get_or_create_session(
        user_id=user_id,
        session_id=session_id,
        get_session_config=run_config.get_session_config,
    )

  from ..agents.base_agent import BaseAgent
  from ..workflow._base_node import BaseNode

  if isinstance(runner.agent, BaseNode) and not isinstance(
      runner.agent, BaseAgent
  ):
    async with aclosing(
        run_node_live(
            runner,
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config,
        )
    ) as agen:
      async for event in agen:
        yield event
    return
  root_agent = runner._require_root_agent()
  invocation_context = new_invocation_context_for_live(
      runner,
      session,
      live_request_queue=live_request_queue,
      run_config=run_config,
  )
  # A streaming tool emits its user-facing events here instead of returning
  # them inline; without a queue those enqueues raise.
  invocation_context._event_queue = asyncio.Queue()

  invocation_context.agent = runner._find_agent_to_run(
      invocation_context.session, root_agent
  )
  if invocation_context.agent and invocation_context.agent is not root_agent:
    runner._restore_branch_from_history(
        invocation_context, invocation_context.agent, root=root_agent
    )

  async def execute(ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    active_agent = ctx.agent
    if not isinstance(active_agent, BaseAgent):
      raise RuntimeError('Live agent execution has no active BaseAgent.')
    async with aclosing(active_agent.run_live(ctx)) as agen:
      async for event in agen:
        yield event

  async with aclosing(
      runner._merge_live_event_streams(
          invocation_context,
          _with_caller_context(
              runner._exec_with_plugin(
                  invocation_context=invocation_context,
                  session=invocation_context.session,
                  execute_fn=execute,
                  is_live_call=True,
              ),
              caller_ctx,
          ),
      )
  ) as agen:
    async for event in agen:
      yield event
