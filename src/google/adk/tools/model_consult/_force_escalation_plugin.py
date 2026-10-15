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

"""Deterministic escalation triggers for `ModelConsultTool`.

`ForceEscalationPlugin` enforces escalation for cases that can be defined
mechanically rather than relying solely on the executor model's judgment:

* `force_first_consult` requires the first model turn of an invocation to call
  the consultation tool via `FunctionCallingConfig(mode=ANY)`.
* `force_on_repeat` forces a consultation on the next model turn when the
  executor repeats an identical tool call with identical arguments `N` times.
* `require_consult_before` refuses chosen tools in `before_tool_callback` until
  the session's consultation counter (`model_consult:{name}:session_uses`) is
  above zero, giving up after `max_refusals` so the agent cannot get stuck.
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...plugins.base_plugin import BasePlugin
from ._model_consult_tool import _SESSION_USES_STATE_KEY_TEMPLATE
from ._model_consult_tool import DEFAULT_TOOL_NAME
from ._model_consult_tool import ModelConsultTool

if TYPE_CHECKING:
  from ...agents.callback_context import CallbackContext
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse
  from ..base_tool import BaseTool
  from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

CONSULT_REQUIRED_STATUS = 'consult_required'
_DEFAULT_FORCE_ESCALATION_PLUGIN_NAME = 'model_consult_force_escalation'
_MAX_TRACKED_INVOCATIONS = 1024


@dataclasses.dataclass
class _InvocationState:
  """Per-invocation tracking state for `ForceEscalationPlugin`."""

  model_calls: int = 0
  consults: int = 0
  forced: int = 0
  refusals: int = 0
  pending_refusals: int = 0
  pending_force: bool = False
  budget_exhausted: bool = False
  active_agent_has_tool: bool = True
  turn_consulted: bool = False
  consulted_before_turn: bool = False
  agent_has_tool: dict[str, bool] = dataclasses.field(default_factory=dict)
  repeats: dict[str, int] = dataclasses.field(default_factory=dict)


def _has_prior_model_turn(callback_context: CallbackContext) -> bool:
  """Returns whether the active agent already ran a model turn in this run."""
  for event in callback_context.session.events:
    if (
        event.invocation_id == callback_context.invocation_id
        and event.author == callback_context.agent_name
        and event.content is not None
        and event.content.role == 'model'
    ):
      return True
  return False


def _signature(name: str, args: Mapping[str, Any] | None) -> str:
  """Builds a deterministic string signature for a tool call and its args."""
  try:
    rendered = json.dumps(
        dict(args) if args else {}, sort_keys=True, default=str
    )
  except (TypeError, ValueError):
    rendered = str(args)
  return f'{name}:{rendered}'


class ForceEscalationPlugin(BasePlugin):
  """Forces `model_consult` calls that a weak executor would otherwise skip.

  Example:
    ```python
    app = App(
        name='demo',
        root_agent=root_agent,
        plugins=[ForceEscalationPlugin()],
    )
    runner = Runner(app=app, session_service=session_service)
    ```

  Attributes:
    name: Unique name of this plugin instance within a runner.
    tool_name: Name of the `ModelConsultTool` instance to force.
  """

  def __init__(
      self,
      *,
      name: str = _DEFAULT_FORCE_ESCALATION_PLUGIN_NAME,
      tool_name: str = DEFAULT_TOOL_NAME,
      force_first_consult: bool = True,
      force_on_repeat: int | None = 3,
      max_forced: int | None = 2,
      require_consult_before: Collection[str] | None = None,
      max_refusals: int | None = 2,
  ) -> None:
    """Initializes the plugin.

    Args:
      name: Plugin name (must be unique within a runner).
      tool_name: The consultation tool's name, if customized on
        `ModelConsultTool`.
      force_first_consult: Require a consult on the first model turn of every
        invocation.
      force_on_repeat: Force a consult once the executor has emitted the same
        tool call with the same arguments this many times. `None` disables loop
        detection.
      max_forced: Cap on forced consults per invocation, so a stubborn loop
        cannot spend the advisor budget indefinitely. `None` for no cap.
      require_consult_before: Tool names that `before_tool_callback` refuses
        until at least one consultation has succeeded in the session. `None` or
        an empty collection disables tool gating.
      max_refusals: Cap on tool refusals per invocation for tools listed in
        `require_consult_before`, so the agent cannot get stuck if it keeps
        calling a gated tool without consulting. `None` for no cap.

    Raises:
      ValueError: If `name` or `tool_name` is empty, `force_on_repeat` is not
        positive, `max_forced` or `max_refusals` is negative, or
        `require_consult_before` is invalid.
    """
    if not name or not name.strip():
      raise ValueError('name must be a non-empty string')
    if not tool_name or not tool_name.strip():
      raise ValueError('tool_name must be a non-empty string')
    if force_on_repeat is not None and (
        isinstance(force_on_repeat, bool)
        or not isinstance(force_on_repeat, int)
        or force_on_repeat <= 0
    ):
      raise ValueError(
          f'force_on_repeat must be positive or None, got {force_on_repeat}'
      )
    if max_forced is not None and (
        isinstance(max_forced, bool)
        or not isinstance(max_forced, int)
        or max_forced < 0
    ):
      raise ValueError(
          f'max_forced must be non-negative or None, got {max_forced}'
      )
    if isinstance(require_consult_before, (str, bytes, bytearray)) or (
        require_consult_before is not None
        and not isinstance(require_consult_before, Iterable)
    ):
      raise ValueError(
          'require_consult_before must be a collection of tool names, not a'
          ' string'
      )
    gated_tools: set[str] = set()
    if require_consult_before is not None:
      for item in require_consult_before:
        if not isinstance(item, str) or not item.strip():
          raise ValueError(
              'require_consult_before entries must be non-empty strings'
          )
        if item == tool_name:
          raise ValueError(
              f'require_consult_before cannot contain tool_name {tool_name!r}'
          )
        gated_tools.add(item)
    if max_refusals is not None and (
        isinstance(max_refusals, bool)
        or not isinstance(max_refusals, int)
        or max_refusals < 0
    ):
      raise ValueError(
          f'max_refusals must be non-negative or None, got {max_refusals}'
      )
    super().__init__(name=name)
    self.tool_name = tool_name
    self._force_first_consult = force_first_consult
    self._force_on_repeat = force_on_repeat
    self._max_forced = max_forced
    self._require_consult_before = frozenset(gated_tools)
    self._max_refusals = max_refusals
    self._states: dict[str, _InvocationState] = {}

  def _state(self, context: CallbackContext) -> _InvocationState:
    state = self._states.get(context.invocation_id)
    if state is not None:
      return state
    state = _InvocationState()
    for event in context.session.events:
      if (
          event.invocation_id != context.invocation_id
          or event.content is None
          or not event.content.parts
      ):
        continue
      for part in event.content.parts:
        fr = part.function_response
        if fr is None or not isinstance(fr.response, Mapping):
          continue
        status = fr.response.get('status')
        if fr.name == self.tool_name:
          if status in ('ok', 'error'):
            state.consults += 1
            consults_meta = fr.response.get('consults')
            if (
                isinstance(consults_meta, Mapping)
                and consults_meta.get('remaining') == 0
            ):
              state.budget_exhausted = True
          elif status == 'limit_reached':
            state.budget_exhausted = True
        elif (
            fr.name in self._require_consult_before
            and status == CONSULT_REQUIRED_STATUS
        ):
          state.refusals += 1
    if len(self._states) >= _MAX_TRACKED_INVOCATIONS:
      self._states.pop(next(iter(self._states)))
    self._states[context.invocation_id] = state
    return state

  def _session_uses(self, context: CallbackContext) -> int:
    key = _SESSION_USES_STATE_KEY_TEMPLATE.format(name=self.tool_name)
    try:
      return max(int(context.state.get(key, 0) or 0), 0)
    except Exception:  # pylint: disable=broad-exception-caught
      return 0

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    """Forces a consultation tool call on the request when a trigger fires.

    Args:
      callback_context: The callback context for the current agent call.
      llm_request: The request being prepared for the executor LLM.

    Returns:
      Always `None`; modifies `llm_request.config.tool_config` in place when
      forcing a consultation call.
    """
    state = self._state(callback_context)
    tools_dict = llm_request.tools_dict or {}
    tool = tools_dict.get(self.tool_name)
    has_tool = tool is not None
    state.active_agent_has_tool = has_tool
    state.agent_has_tool[callback_context.agent_name] = has_tool
    if not has_tool:
      return None  # The agent on this turn does not have the tool.

    state.turn_consulted = False
    state.consulted_before_turn = (
        state.consults > 0 or self._session_uses(callback_context) > 0
    )
    is_first_call = state.model_calls == 0 and not _has_prior_model_turn(
        callback_context
    )
    state.model_calls += 1

    if isinstance(tool, ModelConsultTool) and not tool.has_remaining_budget(
        callback_context
    ):
      state.budget_exhausted = True
    if state.budget_exhausted:
      return None
    if self._max_forced is not None and state.forced >= self._max_forced:
      return None

    wants_first = (
        self._force_first_consult and is_first_call and state.consults == 0
    )
    if not (wants_first or state.pending_force):
      return None

    if llm_request.config is None:
      llm_request.config = types.GenerateContentConfig()
    fc_config = types.FunctionCallingConfig(
        mode=types.FunctionCallingConfigMode.ANY,
        allowed_function_names=[self.tool_name],
    )
    if llm_request.config.tool_config is None:
      llm_request.config.tool_config = types.ToolConfig(
          function_calling_config=fc_config
      )
    else:
      llm_request.config.tool_config = (
          llm_request.config.tool_config.model_copy(
              update={'function_calling_config': fc_config}
          )
      )
    state.forced += 1
    state.pending_force = False
    logger.info(
        '[model_consult] forcing %s (invocation=%s, reason=%s)',
        self.tool_name,
        callback_context.invocation_id,
        'first_consult' if wants_first else 'repeated_tool_call',
    )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    """Refuses gated tools until the session has completed a consultation.

    Args:
      tool: The tool instance about to be executed.
      tool_args: The arguments passed to the tool.
      tool_context: The execution context for the tool call.

    Returns:
      A refusal dictionary with `status` set to `'consult_required'` when the
      tool is gated and no consultation has succeeded yet in the session, or
      `None` to allow the tool call to proceed.
    """
    del tool_args
    if (
        not self._require_consult_before
        or tool.name not in self._require_consult_before
        or tool_context.tool_confirmation is not None
    ):
      return None

    state = self._state(tool_context)
    agent_has_tool = state.agent_has_tool.get(
        tool_context.agent_name, state.active_agent_has_tool
    )
    if not agent_has_tool or state.budget_exhausted:
      return None
    has_consulted = (
        state.consulted_before_turn
        if state.model_calls > 0
        else (state.consults > 0 or self._session_uses(tool_context) > 0)
    )
    if has_consulted:
      return None
    if self._max_refusals is not None and state.refusals >= self._max_refusals:
      return None

    state.refusals += 1
    state.pending_refusals += 1
    logger.info(
        '[model_consult] refusing %s before %s (invocation=%s, refusal=%d/%s)',
        tool.name,
        self.tool_name,
        tool_context.invocation_id,
        state.refusals,
        self._max_refusals,
    )
    return {
        'status': CONSULT_REQUIRED_STATUS,
        'message': (
            f'Tool `{tool.name}` requires consulting `{self.tool_name}` at'
            ' least once in this session before use. Call'
            f' `{self.tool_name}` first, then retry `{tool.name}`.'
        ),
    }

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Tracks consultation outcomes and repeated executor tool calls.

    Args:
      tool: The tool instance that just executed.
      tool_args: The arguments passed to the tool.
      tool_context: The execution context for the tool call.
      result: The dictionary returned by the tool invocation.

    Returns:
      Always `None`; leaves the tool result unmodified.
    """
    state = self._state(tool_context)

    if tool.name == self.tool_name:
      payload = result if isinstance(result, Mapping) else {}
      status = payload.get('status')
      if status == 'ok':
        state.consults += 1
        consults_meta = payload.get('consults')
        if (
            isinstance(consults_meta, Mapping)
            and consults_meta.get('remaining') == 0
        ):
          state.budget_exhausted = True
      elif status == 'error':
        state.consults += 1
      elif status == 'limit_reached':
        # Never force against an exhausted budget: the forced call would come
        # straight back as limit_reached and waste a turn.
        state.budget_exhausted = True
      if status != 'invalid_request':
        state.repeats.clear()
        state.turn_consulted = state.model_calls > 0
      state.pending_force = False
      return None

    if (
        state.pending_refusals > 0
        and isinstance(result, Mapping)
        and result.get('status') == CONSULT_REQUIRED_STATUS
    ):
      state.pending_refusals -= 1
      return None

    agent_has_tool = state.agent_has_tool.get(
        tool_context.agent_name, state.active_agent_has_tool
    )
    if not self._force_on_repeat or not agent_has_tool or state.turn_consulted:
      return None

    signature = _signature(tool.name, tool_args)
    count = state.repeats.get(signature, 0) + 1
    state.repeats[signature] = count
    if count >= self._force_on_repeat:
      state.pending_force = True
      state.repeats[signature] = 0
      logger.info(
          '[model_consult] loop detected: %s called %d times identically',
          tool.name,
          count,
      )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Clears per-invocation tracking state when a run completes.

    Args:
      invocation_context: The context for the completed invocation.
    """
    self._states.pop(invocation_context.invocation_id, None)
    return None

  @override
  async def on_run_error_callback(
      self,
      *,
      invocation_context: InvocationContext,
      error: Exception,
  ) -> None:
    """Clears per-invocation tracking state when a run aborts with an error.

    Args:
      invocation_context: The context for the failed invocation.
      error: The unhandled exception from the run.
    """
    del error
    self._states.pop(invocation_context.invocation_id, None)
    return None
