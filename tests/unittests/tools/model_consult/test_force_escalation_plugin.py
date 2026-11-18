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

"""Unit tests for ForceEscalationPlugin."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
import google.adk.plugins as plugins_pkg
from google.adk.runners import InMemoryRunner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools import model_consult as model_consult_pkg
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.model_consult import ForceEscalationPlugin
from google.adk.tools.model_consult import ModelConsultTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import Field
import pytest


def _text_response(text: str = 'Advisor guidance.') -> LlmResponse:
  return LlmResponse(
      model_version='fake-advisor-001',
      content=types.Content(
          role='model',
          parts=[types.Part(text=text)],
      ),
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=100,
          candidates_token_count=20,
          thoughts_token_count=10,
          total_token_count=130,
      ),
  )


class _FakeLlm(BaseLlm):
  """Deterministic in-memory LLM for executor and advisor tests."""

  model: str = 'fake-llm'
  responses: list[LlmResponse] = Field(default_factory=list)
  requests: list[LlmRequest] = Field(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    self.requests.append(llm_request.model_copy(deep=True))
    idx = len(self.requests) - 1
    if not self.responses:
      yield _text_response()
      return
    yield self.responses[min(idx, len(self.responses) - 1)]


def _user_event(text: str, *, invocation_id: str = 'inv-1') -> Event:
  return Event(
      invocation_id=invocation_id,
      author='user',
      content=types.Content(role='user', parts=[types.Part(text=text)]),
  )


def _make_tool_context(
    events: list[Event] | None = None,
    *,
    invocation_id: str = 'inv-1',
    state: dict[str, Any] | None = None,
) -> ToolContext:
  agent = LlmAgent(
      name='executor',
      model=_FakeLlm(model='fake-executor'),
      instruction='Investigate checkout alerts.',
  )
  session = Session(
      id='sess-1',
      app_name='test-app',
      user_id='user-1',
      state=dict(state or {}),
      events=list(events or [_user_event('go', invocation_id=invocation_id)]),
  )
  invocation_context = InvocationContext(
      invocation_id=invocation_id,
      agent=agent,
      session=session,
      session_service=InMemorySessionService(),
  )
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id='call-1',
  )


def _request(tool: ModelConsultTool | None) -> LlmRequest:
  request = LlmRequest(model='fake-executor', contents=[])
  if tool is not None:
    request.tools_dict = {tool.name: tool}
  return request


def _forced_names(request: LlmRequest) -> list[str] | None:
  if request.config is None:
    return None
  config = request.config.tool_config
  if config is None or config.function_calling_config is None:
    return None
  if config.function_calling_config.mode != types.FunctionCallingConfigMode.ANY:
    return None
  return list(config.function_calling_config.allowed_function_names or [])


def _other_tool(name: str = 'query_logs') -> FunctionTool:
  tool = FunctionTool(func=lambda: 'ok')
  tool.name = name
  return tool


def test_public_exports_match_across_packages():
  """ForceEscalationPlugin is re-exported from model_consult and plugins."""
  assert ForceEscalationPlugin is plugins_pkg.ForceEscalationPlugin
  assert 'ForceEscalationPlugin' in model_consult_pkg.__all__
  assert 'ForceEscalationPlugin' in plugins_pkg.__all__
  assert plugins_pkg.BasePlugin is not None
  plugin = ForceEscalationPlugin()
  assert plugin.name == 'model_consult_force_escalation'
  assert plugin.tool_name == 'model_consult'


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'name': ''}, 'name must be a non-empty string'),
        ({'name': '   '}, 'name must be a non-empty string'),
        ({'tool_name': ''}, 'tool_name must be a non-empty string'),
        ({'tool_name': '  '}, 'tool_name must be a non-empty string'),
        ({'force_on_repeat': 0}, 'force_on_repeat must be positive or None'),
        ({'force_on_repeat': -2}, 'force_on_repeat must be positive or None'),
        (
            {'force_on_repeat': True},
            'force_on_repeat must be positive or None',
        ),
        (
            {'force_on_repeat': False},
            'force_on_repeat must be positive or None',
        ),
        (
            {'force_on_repeat': 1.5},
            'force_on_repeat must be positive or None',
        ),
        ({'max_forced': -1}, 'max_forced must be non-negative or None'),
        ({'max_forced': True}, 'max_forced must be non-negative or None'),
        ({'max_forced': False}, 'max_forced must be non-negative or None'),
        ({'max_forced': 1.5}, 'max_forced must be non-negative or None'),
        (
            {'require_consult_before': 'edit_file'},
            'require_consult_before must be a collection of tool names',
        ),
        (
            {'require_consult_before': b''},
            'require_consult_before must be a collection of tool names',
        ),
        (
            {'require_consult_before': 123},
            'require_consult_before must be a collection of tool names',
        ),
        (
            {'require_consult_before': ['']},
            'require_consult_before entries must be non-empty strings',
        ),
        (
            {'require_consult_before': ['   ']},
            'require_consult_before entries must be non-empty strings',
        ),
        (
            {'require_consult_before': [123]},
            'require_consult_before entries must be non-empty strings',
        ),
        (
            {'require_consult_before': ['model_consult']},
            'require_consult_before cannot contain tool_name',
        ),
        ({'max_refusals': -1}, 'max_refusals must be non-negative or None'),
        ({'max_refusals': True}, 'max_refusals must be non-negative or None'),
        ({'max_refusals': False}, 'max_refusals must be non-negative or None'),
        ({'max_refusals': 1.5}, 'max_refusals must be non-negative or None'),
    ],
)
def test_invalid_init_parameters_raise_value_error(
    kwargs: dict[str, Any], match: str
):
  """Invalid constructor arguments fail fast with ValueError."""
  with pytest.raises(ValueError, match=match):
    ForceEscalationPlugin(**kwargs)


async def test_first_model_turn_is_forced_to_consult():
  """The first model turn of an invocation is forced to call model_consult."""
  plugin = ForceEscalationPlugin()
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  request = _request(tool)

  result = await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )

  assert result is None
  assert _forced_names(request) == ['model_consult']


async def test_second_turn_is_not_forced():
  """Only the first model turn is forced when no loop trigger is pending."""
  plugin = ForceEscalationPlugin(force_on_repeat=None)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  first, second = _request(tool), _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=first)
  await plugin.before_model_callback(callback_context=cb, llm_request=second)

  assert _forced_names(first) == ['model_consult']
  assert _forced_names(second) is None


async def test_nothing_forced_when_agent_lacks_the_tool():
  """Agents that do not have the consultation tool are left untouched."""
  plugin = ForceEscalationPlugin()
  ctx = _make_tool_context()
  request = _request(None)

  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )

  assert _forced_names(request) is None


async def test_force_first_consult_can_be_disabled():
  """Setting force_first_consult=False skips first-turn forcing."""
  plugin = ForceEscalationPlugin(force_first_consult=False)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  request = _request(tool)

  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )

  assert _forced_names(request) is None


async def test_repeated_identical_tool_call_triggers_a_consult():
  """Repeating the same tool call N times forces a consult on the next turn."""
  plugin = ForceEscalationPlugin(force_first_consult=False, force_on_repeat=3)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  not_yet = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=not_yet)
  assert _forced_names(not_yet) is None

  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  forced = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=forced)
  assert _forced_names(forced) == ['model_consult']

  # Once the pending force has been applied, the next turn is not forced again
  # until the repeat threshold is reached anew.
  after_forced = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=after_forced
  )
  assert _forced_names(after_forced) is None


async def test_varied_tool_calls_are_not_a_loop():
  """Calling the same tool with different arguments does not trigger a loop."""
  plugin = ForceEscalationPlugin(force_first_consult=False, force_on_repeat=3)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()

  for service in ('checkout', 'inventory', 'payments'):
    await plugin.after_tool_callback(
        tool=_other_tool(),
        tool_args={'service': service},
        tool_context=ctx,
        result={},
    )

  request = _request(tool)
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )
  assert _forced_names(request) is None


async def test_successful_consult_resets_loop_tracking():
  """A completed consult clears accumulated repeat counts and pending forces."""
  plugin = ForceEscalationPlugin(force_first_consult=False, force_on_repeat=2)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  args = {'service': 'checkout'}

  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  await plugin.after_tool_callback(
      tool=tool, tool_args={}, tool_context=ctx, result={'status': 'ok'}
  )
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )

  request = _request(tool)
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )
  assert _forced_names(request) is None


async def test_consult_before_first_model_turn_suppresses_first_consult_force():
  """If a consult already occurred in the invocation, wants_first is False."""
  plugin = ForceEscalationPlugin(force_first_consult=True, force_on_repeat=None)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()

  await plugin.after_tool_callback(
      tool=tool, tool_args={}, tool_context=ctx, result={'status': 'ok'}
  )
  request = _request(tool)
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )
  assert _forced_names(request) is None


async def test_max_forced_caps_the_spend():
  """Forced consults per invocation never exceed max_forced."""
  plugin = ForceEscalationPlugin(force_on_repeat=1, max_forced=1)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  first = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=first)
  assert _forced_names(first) == ['model_consult']

  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args={}, tool_context=ctx, result={}
  )
  second = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=second)
  assert _forced_names(second) is None


async def test_max_forced_zero_disables_all_forcing():
  """Setting max_forced=0 prevents both first-turn and loop-triggered forces."""
  plugin = ForceEscalationPlugin(
      force_first_consult=True, force_on_repeat=1, max_forced=0
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  first = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=first)
  assert _forced_names(first) is None

  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args={}, tool_context=ctx, result={}
  )
  second = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=second)
  assert _forced_names(second) is None


async def test_exhausted_budget_from_limit_reached_stops_forcing():
  """A limit_reached tool result marks the budget exhausted for the run."""
  plugin = ForceEscalationPlugin(force_on_repeat=1, max_forced=None)
  tool = ModelConsultTool(model=_FakeLlm(), max_uses=1)
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  await plugin.after_tool_callback(
      tool=tool,
      tool_args={},
      tool_context=ctx,
      result={'status': 'limit_reached'},
  )
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args={}, tool_context=ctx, result={}
  )

  request = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=request)
  assert _forced_names(request) is None


async def test_exhausted_budget_from_zero_remaining_stops_forcing():
  """When a successful consult reports remaining=0, no further force is made."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=1, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm(), max_uses=1)
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'check plan'},
      tool_context=ctx,
      result={
          'status': 'ok',
          'consults': {'used_this_turn': 1, 'max_uses': 1, 'remaining': 0},
      },
  )
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args={}, tool_context=ctx, result={}
  )

  request = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=request)
  assert _forced_names(request) is None


async def test_exhausted_session_budget_across_turns_skips_forcing():
  """Reaching session_max_uses in an earlier turn skips forcing later."""
  plugin = ForceEscalationPlugin(force_first_consult=True, max_forced=None)
  tool = ModelConsultTool(model=_FakeLlm(), session_max_uses=1)

  # Turn 1 consumes the single session consult.
  ctx_turn1 = _make_tool_context(invocation_id='inv-1')
  res = await tool.run_async(
      args={'question': 'initial design'}, tool_context=ctx_turn1
  )
  assert res['status'] == 'ok'
  await plugin.after_run_callback(
      invocation_context=ctx_turn1._invocation_context
  )

  # Turn 2 in the same session has a fresh invocation_id, but session_uses == 1.
  ctx_turn2 = _make_tool_context(
      invocation_id='inv-2',
      state=dict(ctx_turn1.session.state),
  )
  request = _request(tool)
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx_turn2._invocation_context),
      llm_request=request,
  )
  assert _forced_names(request) is None


async def test_custom_tool_name_is_forced_and_tracked():
  """Custom tool_name is respected in both before_model and after_tool."""
  plugin = ForceEscalationPlugin(
      tool_name='ask_architect', force_first_consult=True, force_on_repeat=2
  )
  tool = ModelConsultTool(model=_FakeLlm(), name='ask_architect')
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  first = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=first)
  assert _forced_names(first) == ['ask_architect']

  # Calling ask_architect resets any accumulated regular tool repeat count and
  # is not itself counted as a regular tool repeat.
  await plugin.after_tool_callback(
      tool=_other_tool(),
      tool_args={'service': 'checkout'},
      tool_context=ctx,
      result={},
  )
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'check design'},
      tool_context=ctx,
      result={
          'status': 'ok',
          'consults': {'used_this_turn': 1, 'max_uses': 2, 'remaining': 1},
      },
  )
  second = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=second)
  assert _forced_names(second) is None

  # One more regular tool call only reaches count=1 (since ask_architect reset
  # the repeat counter), so the next model turn is not forced.
  await plugin.after_tool_callback(
      tool=_other_tool(),
      tool_args={'service': 'checkout'},
      tool_context=ctx,
      result={},
  )
  third = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=third)
  assert _forced_names(third) is None


async def test_before_model_callback_initializes_none_request_config():
  """before_model_callback creates GenerateContentConfig when config is None."""
  plugin = ForceEscalationPlugin()
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  request = _request(tool)
  request.config = None  # type: ignore[assignment]

  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )

  assert _forced_names(request) == ['model_consult']


async def test_before_model_callback_preserves_existing_tool_config_fields():
  """before_model_callback preserves sibling fields on existing tool_config."""
  plugin = ForceEscalationPlugin()
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  request = _request(tool)
  retrieval_cfg = types.RetrievalConfig(
      lat_lng=types.LatLng(latitude=37.42, longitude=-122.08)
  )
  request.config.tool_config = types.ToolConfig(retrieval_config=retrieval_cfg)

  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )

  assert _forced_names(request) == ['model_consult']
  assert request.config.tool_config is not None
  assert request.config.tool_config.retrieval_config == retrieval_cfg


async def test_after_tool_callback_with_force_on_repeat_none_is_noop():
  """Regular tool calls are ignored when force_on_repeat is None."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()

  for _ in range(5):
    result = await plugin.after_tool_callback(
        tool=_other_tool(),
        tool_args={'service': 'checkout'},
        tool_context=ctx,
        result={},
    )
    assert result is None

  request = _request(tool)
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx._invocation_context),
      llm_request=request,
  )
  assert _forced_names(request) is None


async def test_default_thresholds_force_on_third_repeat_and_cap_at_two_forces():
  """Default force_on_repeat=3 and max_forced=2 govern escalation behavior."""
  plugin = ForceEscalationPlugin()
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  # Force #1: first model turn of the invocation.
  turn1 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn1)
  assert _forced_names(turn1) == ['model_consult']

  # Two identical regular tool calls do not reach default force_on_repeat=3.
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  turn2 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn2)
  assert _forced_names(turn2) is None

  # Third identical tool call triggers Force #2 (reaching default max_forced=2).
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  turn3 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn3)
  assert _forced_names(turn3) == ['model_consult']

  # Another 3 identical tool calls cannot trigger a 3rd force (max_forced=2).
  for _ in range(3):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  turn4 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn4)
  assert _forced_names(turn4) is None


async def test_repeat_counter_resets_to_zero_after_threshold_triggers():
  """Reaching force_on_repeat resets that signature's counter to zero."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  # Two calls trigger the first loop force and reset the counter to 0.
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  first_forced = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=first_forced
  )
  assert _forced_names(first_forced) == ['model_consult']

  # A single additional call (without an intervening consult) must NOT trigger
  # a second force because the counter was reset to 0 when the threshold fired.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  not_forced = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=not_forced
  )
  assert _forced_names(not_forced) is None

  # One more call reaches 2 again and triggers the next force.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  second_forced = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=second_forced
  )
  assert _forced_names(second_forced) == ['model_consult']


async def test_consult_clears_pending_force_from_earlier_repeat_in_same_turn():
  """A consult in the same batch clears a pending force set by earlier tools."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  # The executor also called model_consult in the same parallel tool turn.
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'help'},
      tool_context=ctx,
      result={
          'status': 'ok',
          'consults': {'used_this_turn': 1, 'max_uses': 2, 'remaining': 1},
      },
  )

  next_turn = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=next_turn)
  assert _forced_names(next_turn) is None


async def test_invalid_request_consult_preserves_repeats_and_clears_pending():
  """An invalid_request consult preserves repeats while clearing pending."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  # Two calls to other_tool set pending_force = True; an invalid_request consult
  # in the same batch still clears pending_force.
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool('other_loop'),
        tool_args=args,
        tool_context=ctx,
        result={},
    )
  # 1st call to query_logs (count = 1).
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  # Executor emits an invalid model_consult(question="") -> advisor not called.
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': '   '},
      tool_context=ctx,
      result={'status': 'invalid_request', 'error': 'question is empty'},
  )
  after_invalid = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=after_invalid
  )
  assert _forced_names(after_invalid) is None

  # 2nd call to query_logs reaches count = 2 because invalid_request did not
  # clear state.repeats.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )

  forced = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=forced)
  assert _forced_names(forced) == ['model_consult']


async def test_error_status_and_positive_remaining_do_not_exhaust_budget():
  """status='error' or remaining > 0 leaves the budget open and clears loops."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm(), max_uses=2)
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  # Consult with remaining=1 (boundary > 0) does not mark budget_exhausted.
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'q1'},
      tool_context=ctx,
      result={
          'status': 'ok',
          'consults': {'used_this_turn': 1, 'max_uses': 2, 'remaining': 1},
      },
  )
  # Accumulate 1 repeat before an 'error' consult; status='error' clears the
  # partial repeat count without marking budget_exhausted.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'q2'},
      tool_context=ctx,
      result={'status': 'error', 'error': 'timeout'},
  )
  # Only 1 call after status='error' does not reach force_on_repeat=2 because
  # status='error' cleared the earlier repeat count.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  not_yet_forced = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=not_yet_forced
  )
  assert _forced_names(not_yet_forced) is None

  # A 2nd call reaches force_on_repeat=2 and forces while budget is open.
  await plugin.after_tool_callback(
      tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
  )
  forced = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=forced)
  assert _forced_names(forced) == ['model_consult']


async def test_signature_handles_key_order_custom_objects_and_tool_names():
  """Repeat detection normalizes key order, custom objects, and tool names."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)

  # 1. Different tools with identical args do not count as repeats.
  await plugin.after_tool_callback(
      tool=_other_tool('tool_a'),
      tool_args={'k': 1},
      tool_context=ctx,
      result={},
  )
  await plugin.after_tool_callback(
      tool=_other_tool('tool_b'),
      tool_args={'k': 1},
      tool_context=ctx,
      result={},
  )
  req1 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=req1)
  assert _forced_names(req1) is None

  # 2. Dicts with different key insertion orders match via sort_keys=True.
  await plugin.after_tool_callback(
      tool=_other_tool('search'),
      tool_args={'b': 2, 'a': 1},
      tool_context=ctx,
      result={},
  )
  await plugin.after_tool_callback(
      tool=_other_tool('search'),
      tool_args={'a': 1, 'b': 2},
      tool_context=ctx,
      result={},
  )
  req2 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=req2)
  assert _forced_names(req2) == ['model_consult']

  # 3. Non-JSON-serializable objects serialize via default=str without raising,
  # and distinct custom objects do not collide.
  class _CustomArg:

    def __init__(self, label: str) -> None:
      self.label = label

    def __str__(self) -> str:
      return f'custom:{self.label}'

  await plugin.after_tool_callback(
      tool=_other_tool('custom_tool'),
      tool_args={'item': _CustomArg('alpha')},
      tool_context=ctx,
      result={},
  )
  await plugin.after_tool_callback(
      tool=_other_tool('custom_tool'),
      tool_args={'item': _CustomArg('beta')},
      tool_context=ctx,
      result={},
  )
  req3 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=req3)
  assert _forced_names(req3) is None

  await plugin.after_tool_callback(
      tool=_other_tool('custom_tool'),
      tool_args={'item': _CustomArg('alpha')},
      tool_context=ctx,
      result={},
  )
  req4 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=req4)
  assert _forced_names(req4) == ['model_consult']

  # 4. Mixed int/str keys that fail sort_keys=True fall back to str(args),
  # and distinct unorderable dicts do not collide.
  unorderable_a: Any = {1: 'one', 'a': 'two'}
  unorderable_b: Any = {1: 'one', 'a': 'three'}
  await plugin.after_tool_callback(
      tool=_other_tool('fallback_tool'),
      tool_args=unorderable_a,
      tool_context=ctx,
      result={},
  )
  await plugin.after_tool_callback(
      tool=_other_tool('fallback_tool'),
      tool_args=unorderable_b,
      tool_context=ctx,
      result={},
  )
  req5_distinct = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=req5_distinct
  )
  assert _forced_names(req5_distinct) is None

  await plugin.after_tool_callback(
      tool=_other_tool('fallback_tool'),
      tool_args=unorderable_a,
      tool_context=ctx,
      result={},
  )
  req5 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=req5)
  assert _forced_names(req5) == ['model_consult']


async def test_multi_agent_router_without_tool_does_not_block_specialist():
  """An upstream agent without model_consult does not consume first consult."""
  plugin = ForceEscalationPlugin(
      force_first_consult=True, force_on_repeat=2, max_forced=2
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context(invocation_id='inv-multi')
  cb = CallbackContext(ctx._invocation_context)

  # Upstream router agent does not have model_consult and repeats its own tool.
  router_req = _request(None)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=router_req
  )
  assert _forced_names(router_req) is None
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool('route_lookup'),
        tool_args={'team': 'sre'},
        tool_context=ctx,
        result={},
    )

  # Downstream specialist agent has model_consult on the same invocation_id:
  # its first turn is still treated as the first model call for model_consult.
  specialist_req = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb, llm_request=specialist_req
  )
  assert _forced_names(specialist_req) == ['model_consult']

  # Also verify that with force_first_consult=False, router tool repeats do not
  # falsely force the specialist agent's first turn, AND the specialist agent's
  # own subsequent repeats ARE tracked once it takes over.
  plugin_no_first = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2
  )
  await plugin_no_first.before_model_callback(
      callback_context=cb, llm_request=_request(None)
  )
  for _ in range(2):
    await plugin_no_first.after_tool_callback(
        tool=_other_tool('route_lookup'),
        tool_args={'team': 'sre'},
        tool_context=ctx,
        result={},
    )
  specialist_unforced = _request(tool)
  await plugin_no_first.before_model_callback(
      callback_context=cb, llm_request=specialist_unforced
  )
  assert _forced_names(specialist_unforced) is None

  for _ in range(2):
    await plugin_no_first.after_tool_callback(
        tool=_other_tool('specialist_lookup'),
        tool_args={'service': 'payments'},
        tool_context=ctx,
        result={},
    )
  specialist_loop_forced = _request(tool)
  await plugin_no_first.before_model_callback(
      callback_context=cb, llm_request=specialist_loop_forced
  )
  assert _forced_names(specialist_loop_forced) == ['model_consult']


async def test_state_is_dropped_on_after_run_and_on_run_error_and_fifo_cap():
  """Per-invocation state is cleaned up on completion, error, and FIFO cap."""
  plugin = ForceEscalationPlugin(force_first_consult=True, force_on_repeat=2)
  tool = ModelConsultTool(model=_FakeLlm())
  ctx1 = _make_tool_context(invocation_id='inv-1')
  cb1 = CallbackContext(ctx1._invocation_context)
  ctx2 = _make_tool_context(invocation_id='inv-2')
  cb2 = CallbackContext(ctx2._invocation_context)

  # 1. after_run_callback clears state for inv-1 without affecting inv-2.
  first_inv1 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=first_inv1
  )
  assert _forced_names(first_inv1) == ['model_consult']
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=_request(tool)
  )

  res_after_run = await plugin.after_run_callback(
      invocation_context=ctx1._invocation_context
  )
  assert res_after_run is None

  # inv-1 is forced again as a fresh invocation, while inv-2 is on its 2nd turn.
  reused_inv1 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=reused_inv1
  )
  assert _forced_names(reused_inv1) == ['model_consult']
  second_inv2 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=second_inv2
  )
  assert _forced_names(second_inv2) is None

  # 2. on_run_error_callback clears state for inv-1 without affecting inv-2.
  await plugin.after_tool_callback(
      tool=_other_tool(),
      tool_args={'service': 'checkout'},
      tool_context=ctx2,
      result={},
  )
  res_on_error = await plugin.on_run_error_callback(
      invocation_context=ctx1._invocation_context,
      error=RuntimeError('boom'),
  )
  assert res_on_error is None

  after_err_inv1 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=after_err_inv1
  )
  assert _forced_names(after_err_inv1) == ['model_consult']
  still_active_inv2 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=still_active_inv2
  )
  assert _forced_names(still_active_inv2) is None

  # 3. close() is a no-op so closing one Runner does not wipe sibling inv-2.
  res_close = await plugin.close()
  assert res_close is None

  still_tracked_inv2 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=still_tracked_inv2
  )
  assert _forced_names(still_tracked_inv2) is None

  # 4. FIFO eviction drops the oldest invocation (inv-2) once 1024 newer
  # invocations are tracked, while keeping inv-1.
  ctx_batch = _make_tool_context(invocation_id='inv-batch-0')
  cb_batch = CallbackContext(ctx_batch._invocation_context)
  for i in range(1023):
    ctx_batch._invocation_context.invocation_id = f'inv-batch-{i}'
    await plugin.before_model_callback(
        callback_context=cb_batch, llm_request=_request(tool)
    )

  retained_inv1 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=retained_inv1
  )
  assert _forced_names(retained_inv1) is None

  evicted_inv2 = _request(tool)
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=evicted_inv2
  )
  assert _forced_names(evicted_inv2) == ['model_consult']


async def test_end_to_end_runner_forces_first_consult_and_repeat_loop():
  """InMemoryRunner drives ForceEscalationPlugin across model and tool turns."""
  advisor_llm = _FakeLlm(
      model='fake-advisor',
      responses=[
          _text_response('First consult guidance.'),
          _text_response('Loop recovery guidance.'),
      ],
  )
  consult_tool = ModelConsultTool(model=advisor_llm)

  def query_logs(service: str) -> dict[str, str]:
    """Queries service logs."""
    return {'service': service, 'status': 'timeout'}

  executor_llm = _FakeLlm(
      model='fake-executor',
      responses=[
          # Step 1: forced first consult -> executor calls model_consult.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-1',
                              name='model_consult',
                              args={'question': 'How should I start?'},
                          )
                      )
                  ],
              )
          ),
          # Step 2: not forced -> executor calls query_logs('checkout') #1.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-2',
                              name='query_logs',
                              args={'service': 'checkout'},
                          )
                      )
                  ],
              )
          ),
          # Step 3: not forced -> executor repeats query_logs('checkout') #2.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-3',
                              name='query_logs',
                              args={'service': 'checkout'},
                          )
                      )
                  ],
              )
          ),
          # Step 4: forced due to repeat=2 -> executor calls model_consult.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-4',
                              name='model_consult',
                              args={'question': 'query_logs keeps timing out'},
                          )
                      )
                  ],
              )
          ),
          # Step 5: final answer.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[types.Part(text='Resolved checkout outage.')],
              )
          ),
      ],
  )

  plugin = ForceEscalationPlugin(
      force_first_consult=True,
      force_on_repeat=2,
      max_forced=2,
  )
  agent = LlmAgent(
      name='sre_executor',
      model=executor_llm,
      instruction='Investigate production issues.',
      tools=[consult_tool, query_logs],
  )
  runner = InMemoryRunner(
      app=App(
          name='test_app',
          root_agent=agent,
          plugins=[plugin],
      )
  )
  session = await runner.session_service.create_session(
      app_name='test_app', user_id='user-1'
  )

  events = [
      event
      async for event in runner.run_async(
          user_id='user-1',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part(text='Checkout is failing.')]
          ),
      )
  ]

  assert events[-1].content is not None
  assert events[-1].content.parts[0].text == 'Resolved checkout outage.'
  assert len(executor_llm.requests) == 5
  assert _forced_names(executor_llm.requests[0]) == ['model_consult']
  assert _forced_names(executor_llm.requests[1]) is None
  assert _forced_names(executor_llm.requests[2]) is None
  assert _forced_names(executor_llm.requests[3]) == ['model_consult']
  assert _forced_names(executor_llm.requests[4]) is None
  assert len(advisor_llm.requests) == 2


async def test_single_turn_sub_agent_without_tool_preserves_parent_repeats():
  """Inline single_turn sub-agents without the tool do not clobber parent."""
  advisor_llm = _FakeLlm(model='fake-advisor')
  consult_tool = ModelConsultTool(model=advisor_llm)

  sub_llm = _FakeLlm(
      model='fake-sub',
      responses=[
          _text_response('Sub result 1'),
          _text_response('Sub result 2'),
      ],
  )
  sub_agent = LlmAgent(
      name='single_turn_helper',
      description='Inline single-turn sub-agent without model_consult.',
      mode='single_turn',
      model=sub_llm,
  )

  parent_llm = _FakeLlm(
      model='fake-parent',
      responses=[
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='p-fc-1',
                              name='single_turn_helper',
                              args={'request': 'same query'},
                          )
                      )
                  ],
              )
          ),
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='p-fc-2',
                              name='single_turn_helper',
                              args={'request': 'same query'},
                          )
                      )
                  ],
              )
          ),
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='p-fc-3',
                              name='model_consult',
                              args={'question': 'Sub-agent repeated'},
                          )
                      )
                  ],
              )
          ),
          _text_response('Parent finished.'),
      ],
  )
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      force_on_repeat=2,
      max_forced=2,
  )
  parent_agent = LlmAgent(
      name='parent_with_consult',
      model=parent_llm,
      tools=[consult_tool],
      sub_agents=[sub_agent],
  )
  runner = InMemoryRunner(
      app=App(name='app_single_turn', root_agent=parent_agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name='app_single_turn', user_id='u1'
  )
  async for _ in runner.run_async(
      user_id='u1',
      session_id=session.id,
      new_message=types.Content(role='user', parts=[types.Part(text='hi')]),
  ):
    pass

  assert [_forced_names(r) for r in parent_llm.requests] == [
      None,
      None,
      ['model_consult'],
      None,
  ]
  assert [_forced_names(r) for r in sub_llm.requests] == [None, None]


async def test_parallel_consult_before_sibling_tool_suppresses_repeat_force():
  """A consult finishing before slower sibling tools in one turn clears loop."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False, force_on_repeat=2, max_forced=None
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  args = {'service': 'checkout'}

  # Simulate a model turn where model_consult finishes BEFORE two sibling calls
  # to query_logs in the same parallel tool batch.
  turn1 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn1)
  assert _forced_names(turn1) is None

  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'help'},
      tool_context=ctx,
      result={
          'status': 'ok',
          'consults': {'used_this_turn': 1, 'max_uses': 2, 'remaining': 1},
      },
  )
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )

  # Because model_consult already ran in turn 1, the slower sibling tool calls
  # from that same batch must not force another consult on turn 2.
  turn2 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn2)
  assert _forced_names(turn2) is None

  # On turn 2 (a new model turn), repeating query_logs twice DOES trigger force.
  for _ in range(2):
    await plugin.after_tool_callback(
        tool=_other_tool(), tool_args=args, tool_context=ctx, result={}
    )
  turn3 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn3)
  assert _forced_names(turn3) == ['model_consult']


async def test_hitl_pause_and_resume_does_not_reforce_first_consult():
  """Resuming a paused run on the same invocation_id skips first force."""
  advisor_llm = _FakeLlm(model='fake-advisor')
  consult_tool = ModelConsultTool(model=advisor_llm)

  def dangerous_action(target: str) -> dict[str, str]:
    """Deletes target resource."""
    return {'deleted': target}

  confirm_tool = FunctionTool(func=dangerous_action, require_confirmation=True)

  executor_llm = _FakeLlm(
      model='fake-executor',
      responses=[
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-1',
                              name='model_consult',
                              args={'question': 'should I delete?'},
                          )
                      )
                  ],
              )
          ),
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-2',
                              name='dangerous_action',
                              args={'target': 'db'},
                          )
                      )
                  ],
              )
          ),
          _text_response('Deleted db as confirmed.'),
      ],
  )
  plugin = ForceEscalationPlugin(
      force_first_consult=True,
      force_on_repeat=3,
      max_forced=2,
  )
  agent = LlmAgent(
      name='executor',
      model=executor_llm,
      tools=[consult_tool, confirm_tool],
  )
  runner = InMemoryRunner(
      app=App(name='app_hitl', root_agent=agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name='app_hitl', user_id='u1'
  )
  events1 = [
      event
      async for event in runner.run_async(
          user_id='u1',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part(text='delete db')]
          ),
      )
  ]
  confirm_fc_id = None
  for event in events1:
    for fc in event.get_function_calls():
      if fc.name == 'adk_request_confirmation':
        confirm_fc_id = fc.id
  assert confirm_fc_id is not None

  resume_msg = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=confirm_fc_id,
                  name='adk_request_confirmation',
                  response={'confirmed': True},
              )
          )
      ],
  )
  async for _ in runner.run_async(
      user_id='u1',
      session_id=session.id,
      new_message=resume_msg,
  ):
    pass

  assert [_forced_names(r) for r in executor_llm.requests] == [
      ['model_consult'],
      None,
      None,
  ]


async def test_require_consult_before_refuses_gated_tool_until_consulted():
  """Gated tools return consult_required until a consult succeeds."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()

  read_resp = await plugin.before_tool_callback(
      tool=_other_tool('read_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert read_resp is None

  edit_resp = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert edit_resp is not None
  assert edit_resp['status'] == 'consult_required'
  assert 'model_consult' in edit_resp['message']
  assert 'edit_file' in edit_resp['message']

  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'plan?'},
      tool_context=ctx,
      result={'status': 'ok'},
  )

  after_consult = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert after_consult is None


async def test_require_consult_before_reads_session_uses_across_turns():
  """A nonzero model_consult:{name}:session_uses counter satisfies the gate."""
  plugin = ForceEscalationPlugin(
      tool_name='ask_advisor',
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  ctx = _make_tool_context(
      invocation_id='inv-2',
      state={'model_consult:ask_advisor:session_uses': 1},
  )

  resp = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert resp is None


async def test_require_consult_before_gives_up_after_max_refusals():
  """After max_refusals refusals, the gate gives up so the agent proceeds."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  ctx = _make_tool_context()
  edit_tool = _other_tool('edit_file')

  first = await plugin.before_tool_callback(
      tool=edit_tool, tool_args={'path': 'a.py'}, tool_context=ctx
  )
  second = await plugin.before_tool_callback(
      tool=edit_tool, tool_args={'path': 'a.py'}, tool_context=ctx
  )
  third = await plugin.before_tool_callback(
      tool=edit_tool, tool_args={'path': 'a.py'}, tool_context=ctx
  )

  assert first is not None and first['status'] == 'consult_required'
  assert second is not None and second['status'] == 'consult_required'
  assert third is None


async def test_require_consult_before_with_zero_max_refusals_never_refuses():
  """max_refusals=0 disables before_tool_callback refusals."""
  zero_plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=0,
  )
  ctx_zero = _make_tool_context()
  assert (
      await zero_plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_zero,
      )
      is None
  )


async def test_require_consult_before_with_none_max_refusals_has_no_cap():
  """max_refusals=None refuses gated tool calls without an invocation cap."""
  unlimited_plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=None,
  )
  ctx_unlimited = _make_tool_context()
  for _ in range(5):
    refusal = await unlimited_plugin.before_tool_callback(
        tool=_other_tool('edit_file'),
        tool_args={'path': 'a.py'},
        tool_context=ctx_unlimited,
    )
    assert refusal is not None and refusal['status'] == 'consult_required'


async def test_require_consult_before_skips_when_agent_lacks_consult_tool():
  """Gating does not block an agent without the consultation tool."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx_no_tool = _make_tool_context(invocation_id='inv-multi')
  ctx_no_tool._invocation_context.agent.name = 'helper_without_consult'
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx_no_tool._invocation_context),
      llm_request=_request(None),
  )

  # Even if a specialist with model_consult runs later in the same invocation
  # and sets active_agent_has_tool=True, per-agent tracking remembers that
  # helper_without_consult lacks the tool while specialist_with_consult has it.
  ctx_specialist = _make_tool_context(invocation_id='inv-multi')
  ctx_specialist._invocation_context.agent.name = 'specialist_with_consult'
  await plugin.before_model_callback(
      callback_context=CallbackContext(ctx_specialist._invocation_context),
      llm_request=_request(tool),
  )

  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_no_tool,
      )
      is None
  )
  specialist_refusal = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx_specialist,
  )
  assert (
      specialist_refusal is not None
      and specialist_refusal['status'] == 'consult_required'
  )


async def test_require_consult_before_skips_when_budget_exhausted():
  """Gating does not block a tool once the advisor budget is exhausted."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  ctx_exhausted = _make_tool_context(invocation_id='inv-exhausted')
  await plugin.after_tool_callback(
      tool=ModelConsultTool(model=_FakeLlm()),
      tool_args={'question': 'q'},
      tool_context=ctx_exhausted,
      result={'status': 'limit_reached'},
  )
  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_exhausted,
      )
      is None
  )


@pytest.mark.parametrize('bad_value', ['corrupt', None, -3, 0])
async def test_require_consult_before_handles_corrupt_or_non_positive_state(
    bad_value: Any,
):
  """Corrupted or non-positive session_uses state is treated as zero."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=1,
  )
  ctx = _make_tool_context(
      state={'model_consult:model_consult:session_uses': bad_value}
  )

  refusal = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert refusal is not None and refusal['status'] == 'consult_required'


async def test_advisor_error_satisfies_gate_for_current_invocation():
  """An advisor error unlocks require_consult_before for the current run."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx1 = _make_tool_context(invocation_id='inv-err-1')
  cb1 = CallbackContext(ctx1._invocation_context)

  # Turn 1: executor calls model_consult, which returns status='error'.
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=_request(tool)
  )
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'plan?'},
      tool_context=ctx1,
      result={'status': 'error', 'error': 'Advisor timed out'},
  )

  # Turn 2 of the same invocation: executor follows _ERROR_MESSAGE and calls
  # edit_file with its own judgment -> allowed.
  await plugin.before_model_callback(
      callback_context=cb1, llm_request=_request(tool)
  )
  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx1,
      )
      is None
  )

  # A new invocation in the same session (where session_uses is still 0) still
  # requires consulting once the transient error has passed.
  ctx2 = _make_tool_context(invocation_id='inv-err-2')
  cb2 = CallbackContext(ctx2._invocation_context)
  await plugin.before_model_callback(
      callback_context=cb2, llm_request=_request(tool)
  )
  refusal = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx2,
  )
  assert refusal is not None and refusal['status'] == 'consult_required'


async def test_parallel_consult_and_gated_tool_in_same_turn_refuses_gated():
  """Calling model_consult and a gated tool in one turn refuses the tool."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context(invocation_id='inv-parallel')
  cb = CallbackContext(ctx._invocation_context)
  edit_tool = _other_tool('edit_file')

  # Start Turn 1 before any consultation has occurred.
  await plugin.before_model_callback(
      callback_context=cb, llm_request=_request(tool)
  )

  # Even if model_consult finishes first in the parallel batch and increments
  # both state.consults and session_uses, edit_file emitted in the same turn
  # (before the executor could read the guidance) is still refused.
  ctx.state['model_consult:model_consult:session_uses'] = 1
  await plugin.after_tool_callback(
      tool=tool,
      tool_args={'question': 'plan?'},
      tool_context=ctx,
      result={'status': 'ok'},
  )
  same_turn_refusal = await plugin.before_tool_callback(
      tool=edit_tool,
      tool_args={'path': 'a.py'},
      tool_context=ctx,
  )
  assert (
      same_turn_refusal is not None
      and same_turn_refusal['status'] == 'consult_required'
  )

  # On Turn 2 (after the executor receives the advisor guidance), edit_file is
  # allowed.
  await plugin.before_model_callback(
      callback_context=cb, llm_request=_request(tool)
  )
  assert (
      await plugin.before_tool_callback(
          tool=edit_tool,
          tool_args={'path': 'a.py'},
          tool_context=ctx,
      )
      is None
  )


async def test_normal_tool_returning_consult_required_counts_for_repeats():
  """A non-refused tool returning status='consult_required' still counts."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      force_on_repeat=2,
      max_forced=2,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context(invocation_id='inv-spoof')
  cb = CallbackContext(ctx._invocation_context)
  status_tool = _other_tool('check_status')

  for _ in range(2):
    req = _request(tool)
    await plugin.before_model_callback(callback_context=cb, llm_request=req)
    assert _forced_names(req) is None
    assert (
        await plugin.before_tool_callback(
            tool=status_tool, tool_args={'id': '1'}, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=status_tool,
        tool_args={'id': '1'},
        tool_context=ctx,
        result={'status': 'consult_required'},
    )

  turn3 = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn3)
  assert _forced_names(turn3) == ['model_consult']


async def test_hitl_confirmation_resume_allows_confirmed_gated_tool():
  """Resuming a human-confirmed gated tool call does not re-refuse it."""
  advisor_llm = _FakeLlm(model='fake-advisor')
  consult_tool = ModelConsultTool(model=advisor_llm)
  deleted_targets: list[str] = []

  def delete_db(target: str) -> dict[str, str]:
    """Deletes target database after human confirmation."""
    deleted_targets.append(target)
    return {'deleted': target}

  confirm_tool = FunctionTool(func=delete_db, require_confirmation=True)
  executor_llm = _FakeLlm(
      model='fake-executor',
      responses=[
          # Attempt 1: refused by before_tool_callback (refusal 1/1).
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-del-1',
                              name='delete_db',
                              args={'target': 'prod'},
                          )
                      )
                  ],
              )
          ),
          # Attempt 2: max_refusals=1 reached, so before_tool_callback allows it
          # and apply_confirmation_gate pauses for HITL confirmation.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-del-2',
                              name='delete_db',
                              args={'target': 'prod'},
                          )
                      )
                  ],
              )
          ),
          # After human confirms and delete_db runs, executor attempts a third
          # call (also allowed because hydrated refusals == 1 >= max_refusals)
          # or finishes.
          _text_response('Deleted prod.'),
      ],
  )
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['delete_db'],
      max_refusals=1,
  )
  agent = LlmAgent(
      name='executor',
      model=executor_llm,
      tools=[consult_tool, confirm_tool],
  )
  runner = InMemoryRunner(
      app=App(name='app_hitl_gate', root_agent=agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name='app_hitl_gate', user_id='u1'
  )
  events1 = [
      event
      async for event in runner.run_async(
          user_id='u1',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part(text='drop prod')]
          ),
      )
  ]
  confirm_fc_id = None
  for event in events1:
    for fc in event.get_function_calls():
      if fc.name == 'adk_request_confirmation':
        confirm_fc_id = fc.id
  assert confirm_fc_id is not None
  assert deleted_targets == []

  resume_msg = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=confirm_fc_id,
                  name='adk_request_confirmation',
                  response={'confirmed': True},
              )
          )
      ],
  )
  async for _ in runner.run_async(
      user_id='u1',
      session_id=session.id,
      new_message=resume_msg,
  ):
    pass

  assert deleted_targets == ['prod']


async def test_state_hydration_restores_consults_refusals_and_budget():
  """Hydrating _InvocationState from session events restores prior counters."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      require_consult_before=['edit_file'],
      max_refusals=1,
  )
  # Case 1: prior refusal event in same invocation_id satisfies max_refusals=1.
  refusal_event = Event(
      invocation_id='inv-hydrate-refusal',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part(text='plain text part without function_response'),
              types.Part(
                  function_response=types.FunctionResponse(
                      id='fc-1',
                      name='edit_file',
                      response={'status': 'consult_required'},
                  )
              ),
          ],
      ),
  )
  other_inv_event = Event(
      invocation_id='other-inv',
      author='executor',
      content=types.Content(role='user', parts=[]),
  )
  ctx_refusal = _make_tool_context(
      events=[other_inv_event, refusal_event],
      invocation_id='inv-hydrate-refusal',
  )
  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_refusal,
      )
      is None
  )

  # Case 2: prior limit_reached event in same invocation_id restores
  # budget_exhausted=True.
  limit_event = Event(
      invocation_id='inv-hydrate-limit',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='fc-2',
                      name='model_consult',
                      response={'status': 'limit_reached'},
                  )
              )
          ],
      ),
  )
  ctx_limit = _make_tool_context(
      events=[limit_event],
      invocation_id='inv-hydrate-limit',
  )
  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_limit,
      )
      is None
  )

  # Case 3: prior error consult event in same invocation_id restores consults=1.
  error_event = Event(
      invocation_id='inv-hydrate-err',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='fc-3',
                      name='model_consult',
                      response={'status': 'error'},
                  )
              )
          ],
      ),
  )
  ctx_err = _make_tool_context(
      events=[error_event],
      invocation_id='inv-hydrate-err',
  )
  assert (
      await plugin.before_tool_callback(
          tool=_other_tool('edit_file'),
          tool_args={'path': 'a.py'},
          tool_context=ctx_err,
      )
      is None
  )

  # Case 4: prior invalid_request consult event does not satisfy the gate or
  # mark budget_exhausted.
  invalid_event = Event(
      invocation_id='inv-hydrate-invalid',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='fc-4',
                      name='model_consult',
                      response={'status': 'invalid_request'},
                  )
              )
          ],
      ),
  )
  ctx_invalid = _make_tool_context(
      events=[invalid_event],
      invocation_id='inv-hydrate-invalid',
  )
  invalid_refusal = await plugin.before_tool_callback(
      tool=_other_tool('edit_file'),
      tool_args={'path': 'a.py'},
      tool_context=ctx_invalid,
  )
  assert (
      invalid_refusal is not None
      and invalid_refusal['status'] == 'consult_required'
  )

  # Case 5: prior ok consult with consults.remaining == 0 restores
  # budget_exhausted=True so subsequent repeat triggers do not force.
  plugin_repeat = ForceEscalationPlugin(
      force_first_consult=False,
      force_on_repeat=1,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  zero_remaining_event = Event(
      invocation_id='inv-hydrate-zero-rem',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='fc-5',
                      name='model_consult',
                      response={
                          'status': 'ok',
                          'consults': {'remaining': 0},
                      },
                  )
              )
          ],
      ),
  )
  ctx_zero_rem = _make_tool_context(
      events=[zero_remaining_event],
      invocation_id='inv-hydrate-zero-rem',
  )
  await plugin_repeat.after_tool_callback(
      tool=_other_tool('query_logs'),
      tool_args={'q': 'a'},
      tool_context=ctx_zero_rem,
      result={},
  )
  req_zero_rem = _request(tool)
  await plugin_repeat.before_model_callback(
      callback_context=CallbackContext(ctx_zero_rem._invocation_context),
      llm_request=req_zero_rem,
  )
  assert _forced_names(req_zero_rem) is None


async def test_refused_gated_calls_do_not_accumulate_repeat_count():
  """Refused before_tool_callback calls are ignored by force_on_repeat."""
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      force_on_repeat=3,
      max_forced=2,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  tool = ModelConsultTool(model=_FakeLlm())
  ctx = _make_tool_context()
  cb = CallbackContext(ctx._invocation_context)
  edit_tool = _other_tool('edit_file')
  args = {'path': 'a.py'}

  # Two refused attempts pass through after_tool_callback with consult_required.
  for _ in range(2):
    turn_req = _request(tool)
    await plugin.before_model_callback(
        callback_context=cb, llm_request=turn_req
    )
    assert _forced_names(turn_req) is None
    refusal = await plugin.before_tool_callback(
        tool=edit_tool, tool_args=args, tool_context=ctx
    )
    assert refusal is not None and refusal['status'] == 'consult_required'
    await plugin.after_tool_callback(
        tool=edit_tool, tool_args=args, tool_context=ctx, result=refusal
    )

  # Third attempt gives up on the gate and actually runs edit_file once.
  turn3_req = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn3_req)
  assert _forced_names(turn3_req) is None
  assert (
      await plugin.before_tool_callback(
          tool=edit_tool, tool_args=args, tool_context=ctx
      )
      is None
  )
  await plugin.after_tool_callback(
      tool=edit_tool,
      tool_args=args,
      tool_context=ctx,
      result={'edited': 'a.py'},
  )

  # Because edit_file only actually executed once, turn 4 is NOT forced.
  turn4_req = _request(tool)
  await plugin.before_model_callback(callback_context=cb, llm_request=turn4_req)
  assert _forced_names(turn4_req) is None


async def test_end_to_end_runner_gates_tool_until_consult_and_across_turns():
  """Runner refuses edit_file before consult and allows it across turns."""
  # Setup
  advisor_llm = _FakeLlm(model='fake-advisor')
  consult_tool = ModelConsultTool(model=advisor_llm)
  read_calls: list[str] = []
  edit_calls: list[str] = []

  def read_file(path: str) -> dict[str, str]:
    """Reads a repository file."""
    read_calls.append(path)
    return {'content': f'contents of {path}'}

  def edit_file(path: str) -> dict[str, str]:
    """Applies a patch to a repository file."""
    edit_calls.append(path)
    return {'patched': path}

  executor_llm = _FakeLlm(
      model='fake-executor',
      responses=[
          # Turn 1 (inv 1): read_file allowed before consult.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-read-1',
                              name='read_file',
                              args={'path': 'bug.py'},
                          )
                      )
                  ],
              )
          ),
          # Turn 2 (inv 1): edit_file refused because session_uses == 0.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-edit-1',
                              name='edit_file',
                              args={'path': 'bug.py'},
                          )
                      )
                  ],
              )
          ),
          # Turn 3 (inv 1): executor calls model_consult as instructed.
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-consult-1',
                              name='model_consult',
                              args={'question': 'Verify patch for bug.py'},
                          )
                      )
                  ],
              )
          ),
          # Turn 4 (inv 1): executor retries edit_file -> allowed!
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-edit-2',
                              name='edit_file',
                              args={'path': 'bug.py'},
                          )
                      )
                  ],
              )
          ),
          _text_response('Patched bug.py.'),
          # Invocation 2 in the same session: edit_file allowed immediately!
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id='fc-edit-3',
                              name='edit_file',
                              args={'path': 'followup.py'},
                          )
                      )
                  ],
              )
          ),
          _text_response('Patched followup.py.'),
      ],
  )
  plugin = ForceEscalationPlugin(
      force_first_consult=False,
      force_on_repeat=3,
      require_consult_before=['edit_file'],
      max_refusals=2,
  )
  agent = LlmAgent(
      name='repair_executor',
      model=executor_llm,
      tools=[consult_tool, read_file, edit_file],
  )
  runner = InMemoryRunner(
      app=App(name='repair_app', root_agent=agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name='repair_app', user_id='u1'
  )

  # Act
  events_inv1 = [
      event
      async for event in runner.run_async(
          user_id='u1',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part(text='fix bug.py')]
          ),
      )
  ]

  # Assert
  assert read_calls == ['bug.py']
  assert edit_calls == ['bug.py']
  edit_responses = [
      part.function_response.response
      for event in events_inv1
      if event.content and event.content.parts
      for part in event.content.parts
      if part.function_response and part.function_response.name == 'edit_file'
  ]
  assert len(edit_responses) == 2
  assert edit_responses[0]['status'] == 'consult_required'
  assert edit_responses[1] == {'patched': 'bug.py'}

  # Second invocation in the same session uses the persisted session_uses state.
  async for _ in runner.run_async(
      user_id='u1',
      session_id=session.id,
      new_message=types.Content(
          role='user', parts=[types.Part(text='now fix followup.py')]
      ),
  ):
    pass
  assert edit_calls == ['bug.py', 'followup.py']
