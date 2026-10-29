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

"""ModelConsultTool: mid-generation escalation from executor to advisor."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import dataclasses
import logging
from typing import Any
from typing import TYPE_CHECKING
import weakref

from google.genai import types
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...utils.instructions_utils import inject_session_state
from ..base_tool import BaseTool
from ._advisor import AdvisorError
from ._advisor import AdvisorResult
from ._advisor import call_advisor
from ._advisor import resolve_advisor_llm
from ._advisor import resolve_thinking_level
from ._context import build_advisor_contents
from ._context import ModelConsultContextConfig
from ._prompts import ADVISOR_HANDOFF_TEMPLATE
from ._prompts import ADVISOR_SYSTEM_INSTRUCTION
from ._prompts import CONTEXT_BLOCK_TEMPLATE
from ._prompts import EXECUTOR_INSTRUCTION
from ._prompts import TOOL_DESCRIPTION

if TYPE_CHECKING:
  from ...agents.callback_context import CallbackContext
  from ...events.event import Event
  from ...models.base_llm import BaseLlm
  from ...models.llm_request import LlmRequest
  from ..tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

DEFAULT_ADVISOR_MODEL = 'gemini-3.1-pro-preview'
DEFAULT_TOOL_NAME = 'model_consult'

# `temp:` state is applied to the live session for the duration of an
# invocation and never persisted, and the invocation id in the key ensures the
# turn budget resets on the next turn even if a caller reuses a session object.
_TURN_USES_STATE_KEY_TEMPLATE = 'temp:model_consult:{name}:{invocation_id}:uses'
# Non-`temp:` state persists across turns in the same session so a multi-turn
# conversation cannot exceed `session_max_uses`.
_SESSION_USES_STATE_KEY_TEMPLATE = 'model_consult:{name}:session_uses'

_TOOL_DESCRIPTION_LIMIT = 300

_TURN_LIMIT_MESSAGE = (
    'The advisor consult budget for this turn is exhausted ({max_uses} of'
    ' {max_uses} used). Continue with your own best judgment, reusing the'
    ' guidance you already received.'
)

_SESSION_LIMIT_MESSAGE = (
    'The advisor consult budget for this session is exhausted'
    ' ({session_max_uses} of {session_max_uses} used). Continue with your own'
    ' best judgment, reusing the guidance you already received.'
)

_ERROR_MESSAGE = (
    'The advisor could not be reached. Continue with your own best judgment;'
    ' do not retry this tool for the same question.'
)


@dataclasses.dataclass
class _SessionConsultState:
  """Per-session concurrency and state-delta coordination for parallel calls."""

  cond: asyncio.Condition = dataclasses.field(default_factory=asyncio.Condition)
  session_ref: Any = None
  active_calls: int = 0
  reserved_session: int = 0
  reserved_turn: dict[str, int] = dataclasses.field(default_factory=dict)
  inv_deltas: dict[str, list[dict[str, Any]]] = dataclasses.field(
      default_factory=dict
  )


def _extract_static_instruction_texts(value: Any) -> list[str]:
  """Extracts text segments from a `types.ContentUnion` static instruction."""
  if isinstance(value, str):
    return [value]
  if isinstance(value, types.Part):
    return [value.text] if value.text else []
  if isinstance(value, types.Content):
    return [
        text
        for part in value.parts or []
        for text in _extract_static_instruction_texts(part)
    ]
  if isinstance(value, Sequence) and not isinstance(
      value, (str, bytes, bytearray)
  ):
    return [
        text
        for item in value
        for text in _extract_static_instruction_texts(item)
    ]
  return []


class ModelConsultTool(BaseTool):
  """Lets an executor agent consult a stronger advisor model mid-generation.

  The advisor reads the executor's full session -- instructions, reasoning,
  tool calls and tool results -- and returns a plan or course correction. The
  executor keeps doing the work, so the bulk of token generation stays at
  executor rates.

  Example:
    ```python
    root_agent = Agent(
        model='gemini-3.5-flash',
        name='root_cause_analysis_agent',
        instruction='...',
        tools=[
            ModelConsultTool(
                model='gemini-3.1-pro-preview',
                max_uses=2,
                session_max_uses=5,
            )
        ],
    )
    ```

  Attributes:
    advisor_model: The resolved advisor `BaseLlm`.
    max_uses: Consults allowed per turn, or `None` for unlimited.
    session_max_uses: Consults allowed across the entire session, or `None` for
      unlimited.
    max_output_tokens: Output token cap for the advisor response, if configured.
    thinking_level: Normalized advisor thinking level (`minimal`, `low`,
      `medium`, `high`), or `None`.
  """

  def __init__(
      self,
      *,
      model: str | BaseLlm = DEFAULT_ADVISOR_MODEL,
      max_uses: int | None = None,
      session_max_uses: int | None = None,
      thinking_level: str | types.ThinkingLevel | None = 'high',
      max_output_tokens: int | None = None,
      name: str = DEFAULT_TOOL_NAME,
      description: str | None = None,
      executor_instruction: str | None = None,
      advisor_instruction: str | None = None,
      include_agent_instruction: bool = True,
      include_tool_inventory: bool = True,
      context_config: ModelConsultContextConfig | None = None,
      generate_content_config: types.GenerateContentConfig | None = None,
      timeout_seconds: float | None = None,
  ):
    """Initializes the tool.

    Args:
      model: Advisor model name (resolved through ADK's model registry) or a
        `BaseLlm` instance, so any model ADK supports can advise.
      max_uses: Maximum advisor consults allowed in a single turn. `None` (the
        default) means unlimited.
      session_max_uses: Maximum advisor consults allowed across the entire
        session. `None` (the default) means unlimited. When both `max_uses` and
        `session_max_uses` are set, whichever limit is reached first blocks
        further consults.
      thinking_level: `'minimal'`, `'low'`, `'medium'`, `'high'` (default), or
        a `types.ThinkingLevel` enum value. `None` disables overriding the
        thinking config.
      max_output_tokens: Caps the advisor's output (thinking included on models
        that bill it there). Advisor output is the single largest cost driver of
        this pattern, so a cap is the cheapest lever available -- but a cap that
        is too tight starves a high thinking_level and returns nothing. Measure
        before setting it below ~4096 with `thinking_level='high'`.
      name: Tool name the executor sees. Change it only if it collides.
      description: Overrides the tuned tool description that steers escalation.
      executor_instruction: Overrides the escalation policy automatically
        appended to the executor's `system_instruction`. Pass `""` to disable
        automatic injection.
      advisor_instruction: Overrides the advisor's system instruction.
      include_agent_instruction: Forward the executor agent's own instruction to
        the advisor, so guidance respects the executor's constraints.
      include_tool_inventory: Tell the advisor which tools the executor can
        call, so the plan names real tools with real arguments instead of steps
        the executor cannot perform.
      context_config: How much session context to hand over.
      generate_content_config: Extra generation config for the advisor call
        (temperature, max_output_tokens, safety settings...).
      timeout_seconds: Abort the advisor call after this long. On timeout the
        executor is told to proceed on its own rather than failing the turn.

    Raises:
      ValueError: If `max_uses`, `session_max_uses`, `max_output_tokens`, or
        `timeout_seconds` is not positive, or if `thinking_level` or `model` is
        invalid.
    """
    super().__init__(
        name=name,
        description=description or TOOL_DESCRIPTION,
    )
    if max_uses is not None and max_uses <= 0:
      raise ValueError(f'max_uses must be positive or None, got {max_uses}')
    if session_max_uses is not None and session_max_uses <= 0:
      raise ValueError(
          f'session_max_uses must be positive or None, got {session_max_uses}'
      )
    cfg_max_output_tokens = (
        generate_content_config.max_output_tokens
        if generate_content_config is not None
        else None
    )
    if max_output_tokens is not None and max_output_tokens <= 0:
      raise ValueError(
          f'max_output_tokens must be positive or None, got {max_output_tokens}'
      )
    if cfg_max_output_tokens is not None and cfg_max_output_tokens <= 0:
      raise ValueError(
          'generate_content_config.max_output_tokens must be positive or None,'
          f' got {cfg_max_output_tokens}'
      )
    if (
        max_output_tokens is not None
        and cfg_max_output_tokens is not None
        and max_output_tokens != cfg_max_output_tokens
    ):
      raise ValueError(
          f'Conflicting max_output_tokens ({max_output_tokens}) and'
          ' generate_content_config.max_output_tokens'
          f' ({cfg_max_output_tokens})'
      )
    if timeout_seconds is not None and timeout_seconds <= 0:
      raise ValueError(
          f'timeout_seconds must be positive or None, got {timeout_seconds}'
      )

    self.advisor_model = resolve_advisor_llm(model)
    self.max_output_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else cfg_max_output_tokens
    )
    self.max_uses = max_uses
    self.session_max_uses = session_max_uses
    self._thinking_level = resolve_thinking_level(thinking_level)
    self.thinking_level: str | None = (
        self._thinking_level.value.lower()
        if self._thinking_level is not None
        else None
    )
    self._advisor_instruction = (
        advisor_instruction or ADVISOR_SYSTEM_INSTRUCTION
    )
    self._include_agent_instruction = include_agent_instruction
    self._include_tool_inventory = include_tool_inventory
    self._context_config = context_config or ModelConsultContextConfig()
    if max_output_tokens is not None and cfg_max_output_tokens is None:
      generate_content_config = (
          generate_content_config.model_copy(deep=True)
          if generate_content_config is not None
          else types.GenerateContentConfig()
      )
      generate_content_config.max_output_tokens = max_output_tokens
    self._generate_content_config = generate_content_config
    self._timeout_seconds = timeout_seconds
    if executor_instruction is not None:
      self._executor_system_instruction = executor_instruction.strip()
    elif self.name == DEFAULT_TOOL_NAME:
      self._executor_system_instruction = EXECUTOR_INSTRUCTION
    else:
      self._executor_system_instruction = EXECUTOR_INSTRUCTION.replace(
          f'`{DEFAULT_TOOL_NAME}`', f'`{self.name}`'
      )
    self._session_states: dict[int, _SessionConsultState] = {}

  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'question': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'The specific decision or blocker you want reviewed.'
                        ' State the approach you are considering, or what you'
                        ' tried and how it failed. Be concrete; the advisor'
                        ' already sees the conversation, so do not restate it.'
                    ),
                ),
                'context': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'Optional. Anything material that is NOT visible in the'
                        ' conversation: constraints you inferred, observations'
                        ' from outside this session, or the options you are'
                        ' weighing.'
                    ),
                ),
            },
            required=['question'],
        ),
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
    if not self._executor_system_instruction:
      return
    existing = llm_request.config.system_instruction
    if (
        not isinstance(existing, str)
        or self._executor_system_instruction not in existing
    ):
      llm_request.append_instructions([self._executor_system_instruction])

  def _turn_uses_state_key(
      self, tool_context: ToolContext | CallbackContext
  ) -> str:
    return _TURN_USES_STATE_KEY_TEMPLATE.format(
        name=self.name,
        invocation_id=tool_context.invocation_id or 'unknown',
    )

  def _session_uses_state_key(self) -> str:
    return _SESSION_USES_STATE_KEY_TEMPLATE.format(name=self.name)

  def _read_state_counter(
      self, tool_context: ToolContext | CallbackContext, key: str
  ) -> int:
    try:
      return max(int(tool_context.state.get(key, 0) or 0), 0)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.debug(
          'ModelConsultTool could not read state counter %s',
          key,
          exc_info=True,
      )
      return 0

  def _turn_uses_so_far(
      self, tool_context: ToolContext | CallbackContext
  ) -> int:
    return self._read_state_counter(
        tool_context, self._turn_uses_state_key(tool_context)
    )

  def _session_uses_so_far(
      self, tool_context: ToolContext | CallbackContext
  ) -> int:
    return self._read_state_counter(
        tool_context, self._session_uses_state_key()
    )

  def has_remaining_budget(
      self, context: ToolContext | CallbackContext
  ) -> bool:
    """Returns whether at least one consult remains in the turn and session.

    Args:
      context: The tool or callback context holding the session state.

    Returns:
      True if neither `max_uses` nor `session_max_uses` has been reached.
    """
    if (
        self.session_max_uses is not None
        and self._session_uses_so_far(context) >= self.session_max_uses
    ):
      return False
    if (
        self.max_uses is not None
        and self._turn_uses_so_far(context) >= self.max_uses
    ):
      return False
    return True

  def _write_state_counter(
      self, tool_context: ToolContext, key: str, value: int
  ) -> None:
    try:
      tool_context.state[key] = value
    except TypeError:
      # Fallback to the underlying `session.state` dict (`State._value`) if
      # `tool_context.state` rejects item assignment (e.g., a custom state
      # mapping or a `state_schema` validator that only exempts `app:`/`user:`/
      # `temp:` prefixes), while `_record_use` updates `actions.state_delta`.
      tool_context.session.state[key] = value

  def _record_use(
      self,
      tool_context: ToolContext,
      *,
      turn_uses: int,
      session_uses: int,
      deltas: list[dict[str, Any]],
  ) -> None:
    turn_key = self._turn_uses_state_key(tool_context)
    session_key = self._session_uses_state_key()
    try:
      self._write_state_counter(tool_context, turn_key, turn_uses)
      self._write_state_counter(tool_context, session_key, session_uses)
      for delta in deltas:
        delta[turn_key] = max(int(delta.get(turn_key, 0) or 0), turn_uses)
        delta[session_key] = max(
            int(delta.get(session_key, 0) or 0), session_uses
        )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.warning(
          'ModelConsultTool could not persist its use counters', exc_info=True
      )

  def _strip_executor_escalation_instruction(self, text: str) -> str:
    for snippet in (self._executor_system_instruction, EXECUTOR_INSTRUCTION):
      if snippet and snippet in text:
        text = text.replace(snippet, '')
    return text.strip()

  async def _executor_instruction(
      self, tool_context: ToolContext
  ) -> str | None:
    """Best-effort read of the executor agent's own instruction."""
    if not self._include_agent_instruction:
      return None
    invocation_context = tool_context._invocation_context
    agent: Any = invocation_context.agent
    canonical: Any = getattr(agent, 'canonical_instruction', None)
    if not callable(canonical):
      return None

    parts: list[str] = []
    static_inst: Any = getattr(agent, 'static_instruction', None)
    if static_inst:
      static_lines = [
          self._strip_executor_escalation_instruction(text)
          for text in _extract_static_instruction_texts(static_inst)
      ]
      static_lines = [line for line in static_lines if line]
      if static_lines:
        parts.append('\n'.join(static_lines))

    readonly_ctx = ReadonlyContext(invocation_context)
    try:
      instruction, bypass_state_injection = await canonical(readonly_ctx)
      if instruction and not bypass_state_injection:
        try:
          instruction = await inject_session_state(instruction, readonly_ctx)
        except Exception:  # pylint: disable=broad-exception-caught
          logger.debug(
              'ModelConsultTool could not inject session state into'
              ' instruction',
              exc_info=True,
          )
          for key, val in invocation_context.session.state.items():
            if isinstance(key, str) and key.isidentifier():
              replacement = '' if val is None else str(val)
              instruction = instruction.replace(f'{{{key}}}', replacement)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.debug(
          'ModelConsultTool could not read the agent instruction', exc_info=True
      )
      return '\n\n'.join(parts) or None
    instruction = self._strip_executor_escalation_instruction(instruction or '')
    if instruction:
      parts.append(instruction)
    return '\n\n'.join(parts) or None

  async def _tool_inventory(self, tool_context: ToolContext) -> str | None:
    """Lists the executor's other tools so guidance can name them."""
    if not self._include_tool_inventory:
      return None
    invocation_context = tool_context._invocation_context
    tools = invocation_context.canonical_tools_cache
    if tools is None:
      agent: Any = invocation_context.agent
      canonical_tools: Any = getattr(agent, 'canonical_tools', None)
      if not callable(canonical_tools):
        return None
      try:
        tools = await canonical_tools(ReadonlyContext(invocation_context))
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug(
            "ModelConsultTool could not read the agent's tools", exc_info=True
        )
        return None
      invocation_context.canonical_tools_cache = tools

    lines: list[str] = []
    for tool in tools:
      if tool.name == self.name:
        continue
      description = ' '.join((tool.description or '').split())
      if len(description) > _TOOL_DESCRIPTION_LIMIT:
        description = description[:_TOOL_DESCRIPTION_LIMIT] + '...'
      lines.append(
          f'- {tool.name}: {description}' if description else f'- {tool.name}'
      )
    return '\n'.join(lines) or None

  def _normalized_agent_name(self, tool_context: ToolContext) -> str | None:
    raw_name: Any = tool_context.agent_name
    if not isinstance(raw_name, str):
      return None
    name = raw_name.strip()
    return name if name and name != 'unknown' else None

  def _handoff_part(
      self, question: str, context: str | None, agent_name: str | None
  ) -> types.Part:
    context_block = (
        CONTEXT_BLOCK_TEMPLATE.format(context=context.strip())
        if context and context.strip()
        else ''
    )
    agent_clause = f' ({agent_name})' if agent_name else ''
    text = ADVISOR_HANDOFF_TEMPLATE.format(
        agent_clause=agent_clause,
        question=question.strip(),
        context_block=context_block,
    )
    return types.Part(text=text)

  def _in_flight_consult_call_ids(self, events: list[Event]) -> list[str]:
    """Returns unanswered model_consult function call ids in the event log."""
    answered_ids: set[str] = set()
    consult_call_ids: list[str] = []
    for event in events:
      if event.content is None or not event.content.parts:
        continue
      for part in event.content.parts:
        fr = part.function_response
        if fr is not None and fr.id:
          answered_ids.add(fr.id)
        fc = part.function_call
        if fc is not None and fc.name == self.name and fc.id:
          consult_call_ids.append(fc.id)
    return [cid for cid in consult_call_ids if cid not in answered_ids]

  def _build_contents(
      self,
      tool_context: ToolContext,
      question: str,
      context: str | None,
      *,
      executor_instruction: str | None = None,
      tool_inventory: str | None = None,
  ) -> list[types.Content]:
    agent_name = self._normalized_agent_name(tool_context)
    parts: list[types.Part] = []
    if executor_instruction:
      agent_label = agent_name or 'the executor'
      parts.append(
          types.Part(
              text=(
                  '--- EXECUTOR AGENT INSTRUCTION'
                  f' ({agent_label}) ---\nThe executor operates under the'
                  ' following instruction. Your guidance must respect'
                  f' it.\n\n{executor_instruction}'
              )
          )
      )
    if tool_inventory:
      parts.append(
          types.Part(
              text=(
                  '--- TOOLS AVAILABLE TO THE EXECUTOR ---\nThese are the only'
                  ' tools the executor can call. Name them explicitly in your'
                  ' plan, with concrete arguments. Do not propose steps that'
                  f' require tools not listed here.\n\n{tool_inventory}'
              )
          )
      )

    events = list(tool_context.session.events)
    in_flight_consult_ids = self._in_flight_consult_call_ids(events)
    skip_ids = [
        call_id
        for call_id in (
            tool_context.function_call_id,
            *in_flight_consult_ids,
        )
        if call_id
    ]
    session_contents = build_advisor_contents(
        events, config=self._context_config, skip_function_call_ids=skip_ids
    )
    for content in session_contents:
      parts.extend(content.parts or [])

    parts.append(self._handoff_part(question, context, agent_name))
    return [types.Content(role='user', parts=parts)]

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    """Consults the advisor and returns its guidance.

    Args:
      args: Tool call arguments (`question` and optional `context`).
      tool_context: The execution context for the current tool call.

    Returns:
      A structured dictionary with `status` set to `'ok'`, `'limit_reached'`,
      `'error'`, or `'invalid_request'`. Never raises: budget exhaustion and
      advisor failures return a response the executor can read and continue
      from.
    """
    raw_question = args.get('question')
    question = (
        raw_question.strip()
        if isinstance(raw_question, str)
        else str(raw_question or '').strip()
    )
    if not question:
      return {
          'status': 'invalid_request',
          'message': (
              '`question` is required: state the decision or blocker you want'
              ' reviewed.'
          ),
      }

    session = tool_context.session
    session_id_key = id(session)
    inv_key = tool_context.invocation_id or 'unknown'
    state = self._session_states.get(session_id_key)
    if state is not None and state.session_ref is not None:
      if state.session_ref() is not session:
        state = None
    if state is None:
      state = _SessionConsultState(session_ref=weakref.ref(session))
      self._session_states[session_id_key] = state
      weakref.finalize(session, self._session_states.pop, session_id_key, None)
    state_delta = tool_context.actions.state_delta
    reserved = False

    try:
      async with state.cond:
        state.active_calls += 1
        deltas_for_inv = state.inv_deltas.setdefault(inv_key, [])
        if not any(existing is state_delta for existing in deltas_for_inv):
          deltas_for_inv.append(state_delta)

        while True:
          turn_uses = self._turn_uses_so_far(tool_context)
          session_uses = self._session_uses_so_far(tool_context)

          if (
              self.session_max_uses is not None
              and session_uses >= self.session_max_uses
          ):
            logger.info(
                'ModelConsultTool session budget exhausted (%s/%s)',
                session_uses,
                self.session_max_uses,
            )
            return {
                'status': 'limit_reached',
                'message': _SESSION_LIMIT_MESSAGE.format(
                    session_max_uses=self.session_max_uses
                ),
                'consults': self._consult_stats(turn_uses, session_uses),
            }

          if self.max_uses is not None and turn_uses >= self.max_uses:
            logger.info(
                'ModelConsultTool turn budget exhausted for invocation %s'
                ' (%s/%s)',
                tool_context.invocation_id or '?',
                turn_uses,
                self.max_uses,
            )
            return {
                'status': 'limit_reached',
                'message': _TURN_LIMIT_MESSAGE.format(max_uses=self.max_uses),
                'consults': self._consult_stats(turn_uses, session_uses),
            }

          turn_reserved = state.reserved_turn.get(inv_key, 0)
          session_saturated = (
              self.session_max_uses is not None
              and session_uses + state.reserved_session >= self.session_max_uses
          )
          turn_saturated = (
              self.max_uses is not None
              and turn_uses + turn_reserved >= self.max_uses
          )
          if not session_saturated and not turn_saturated:
            state.reserved_session += 1
            state.reserved_turn[inv_key] = turn_reserved + 1
            reserved = True
            break
          await state.cond.wait()

      raw_context = args.get('context')
      extra_context = (
          raw_context.strip()
          if isinstance(raw_context, str)
          else (str(raw_context).strip() if raw_context is not None else None)
      )
      executor_instruction = await self._executor_instruction(tool_context)
      tool_inventory = await self._tool_inventory(tool_context)
      contents = self._build_contents(
          tool_context,
          question,
          extra_context,
          executor_instruction=executor_instruction,
          tool_inventory=tool_inventory,
      )

      try:
        result = await call_advisor(
            self.advisor_model,
            contents=contents,
            system_instruction=self._advisor_instruction,
            thinking_level=self._thinking_level,
            generate_content_config=self._generate_content_config,
            timeout_seconds=self._timeout_seconds,
            agent_name=self.name,
        )
      except AdvisorError as exc:
        logger.warning('ModelConsultTool advisor call failed: %s', exc)
        async with state.cond:
          turn_uses = self._turn_uses_so_far(tool_context)
          session_uses = self._session_uses_so_far(tool_context)
          return {
              'status': 'error',
              'error': str(exc),
              'message': _ERROR_MESSAGE,
              'advisor_model': self.advisor_model.model,
              'consults': self._consult_stats(turn_uses, session_uses),
          }

      async with state.cond:
        turn_uses = self._turn_uses_so_far(tool_context) + 1
        session_uses = self._session_uses_so_far(tool_context) + 1
        self._record_use(
            tool_context,
            turn_uses=turn_uses,
            session_uses=session_uses,
            deltas=state.inv_deltas.get(inv_key, []),
        )
        return self._success_payload(result, turn_uses, session_uses)
    finally:
      async with state.cond:
        if reserved:
          state.reserved_session = max(state.reserved_session - 1, 0)
          remaining_reserved = state.reserved_turn.get(inv_key, 1) - 1
          if remaining_reserved <= 0:
            state.reserved_turn.pop(inv_key, None)
          else:
            state.reserved_turn[inv_key] = remaining_reserved
          state.cond.notify_all()
        state.active_calls = max(state.active_calls - 1, 0)
        if state.active_calls == 0:
          state.inv_deltas.clear()

  def _consult_stats(self, turn_uses: int, session_uses: int) -> dict[str, Any]:
    turn_remaining = (
        None if self.max_uses is None else max(self.max_uses - turn_uses, 0)
    )
    session_remaining = (
        None
        if self.session_max_uses is None
        else max(self.session_max_uses - session_uses, 0)
    )
    if turn_remaining is not None and session_remaining is not None:
      remaining: int | None = min(turn_remaining, session_remaining)
    elif turn_remaining is not None:
      remaining = turn_remaining
    else:
      remaining = session_remaining

    return {
        'used_this_turn': turn_uses,
        'max_uses': self.max_uses,
        'used_this_session': session_uses,
        'session_max_uses': self.session_max_uses,
        'remaining': remaining,
    }

  def _success_payload(
      self, result: AdvisorResult, turn_uses: int, session_uses: int
  ) -> dict[str, Any]:
    return {
        'status': 'ok',
        'guidance': result.text,
        'advisor_model': result.model_version or result.model,
        'thinking_level': self.thinking_level,
        'consults': self._consult_stats(turn_uses, session_uses),
        'usage': result.usage.to_dict(),
        'latency_ms': result.latency_ms,
    }
