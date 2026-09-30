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

"""Unit tests for ModelConsultTool."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.functions import merge_parallel_function_response_events
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from google.adk.tools import model_consult as model_consult_pkg
from google.adk.tools import ModelConsultContextConfig as TopLevelContextConfig
from google.adk.tools import ModelConsultTool as TopLevelModelConsultTool
from google.adk.tools.model_consult import ADVISOR_SYSTEM_INSTRUCTION
from google.adk.tools.model_consult import ContextMode
from google.adk.tools.model_consult import DEFAULT_ADVISOR_MODEL
from google.adk.tools.model_consult import DEFAULT_TOOL_NAME
from google.adk.tools.model_consult import EXECUTOR_INSTRUCTION
from google.adk.tools.model_consult import ModelConsultContextConfig
from google.adk.tools.model_consult import ModelConsultTool
from google.adk.tools.model_consult import TOOL_DESCRIPTION
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
import pytest


def _text_response(
    text: str = '1. Diagnosis. 2. Plan. 3. Watch out.',
    *,
    model_version: str | None = 'fake-advisor-001',
    prompt_tokens: int = 1000,
    output_tokens: int = 120,
    thoughts_tokens: int = 50,
) -> LlmResponse:
  return LlmResponse(
      model_version=model_version,
      content=types.Content(
          role='model',
          parts=[types.Part(text=text)],
      ),
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=prompt_tokens,
          candidates_token_count=output_tokens,
          thoughts_token_count=thoughts_tokens,
          total_token_count=prompt_tokens + output_tokens + thoughts_tokens,
      ),
  )


class _FakeAdvisorLlm(BaseLlm):
  """Deterministic in-memory advisor LLM for tool tests."""

  model: str = 'fake-advisor'
  responses: list[LlmResponse] = Field(default_factory=list)
  errors: list[Exception | None] = Field(default_factory=list)
  requests: list[LlmRequest] = Field(default_factory=list)
  delay_seconds: float = 0.0
  per_call_delays: list[float] = Field(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    self.requests.append(llm_request.model_copy(deep=True))
    call_idx = len(self.requests) - 1
    delay = (
        self.per_call_delays[call_idx]
        if call_idx < len(self.per_call_delays)
        else self.delay_seconds
    )
    if delay > 0:
      await asyncio.sleep(delay)
    if call_idx < len(self.errors) and self.errors[call_idx] is not None:
      raise self.errors[call_idx]
    if not self.responses:
      yield _text_response()
      return
    response = self.responses[min(call_idx, len(self.responses) - 1)]
    yield response


def _user_event(text: str) -> Event:
  return Event(
      invocation_id='inv-1',
      author='user',
      content=types.Content(role='user', parts=[types.Part(text=text)]),
  )


def _agent_event(parts: list[types.Part], *, author: str = 'executor') -> Event:
  return Event(
      invocation_id='inv-1',
      author=author,
      content=types.Content(role='model', parts=parts),
  )


def _tool_result_event(
    name: str,
    response: dict[str, Any],
    *,
    call_id: str = 'fc-1',
    author: str = 'executor',
) -> Event:
  return Event(
      invocation_id='inv-1',
      author=author,
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name=name, response=response
                  )
              )
          ],
      ),
  )


def _make_tool_context(
    events: list[Event] | None = None,
    *,
    instruction: str = 'Investigate production issues carefully.',
    static_instruction: types.ContentUnion | None = None,
    tools: list[Any] | None = None,
    session: Session | None = None,
    invocation_id: str = 'inv-1',
    function_call_id: str | None = 'fc-consult',
) -> ToolContext:
  agent = LlmAgent(
      name='executor',
      model='gemini-2.5-flash',
      instruction=instruction,
      static_instruction=static_instruction,
      tools=tools or [],
  )
  if session is None:
    session = Session(
        id='session-1',
        app_name='test-app',
        user_id='user-1',
        state={},
        events=list(events or []),
    )
  elif events is not None:
    session.events = list(events)
  invocation_context = InvocationContext(
      session_service=InMemorySessionService(),
      invocation_id=invocation_id,
      agent=agent,
      session=session,
  )
  return ToolContext(
      invocation_context,
      function_call_id=function_call_id,
  )


async def _run(
    tool: ModelConsultTool, tool_context: ToolContext, **args: Any
) -> dict[str, Any]:
  return await tool.run_async(args=args, tool_context=tool_context)


def test_public_exports_and_prompt_constants():
  """Verifies public re-exports on tools and model_consult packages."""
  assert TopLevelModelConsultTool is ModelConsultTool
  assert TopLevelContextConfig is ModelConsultContextConfig
  expected_all = {
      'ADVISOR_SYSTEM_INSTRUCTION',
      'ContextMode',
      'DEFAULT_ADVISOR_MODEL',
      'DEFAULT_TOOL_NAME',
      'EXECUTOR_INSTRUCTION',
      'ModelConsultContextConfig',
      'ModelConsultTool',
      'TOOL_DESCRIPTION',
  }
  assert set(model_consult_pkg.__all__) == expected_all
  assert ContextMode is not None
  assert DEFAULT_TOOL_NAME == 'model_consult'
  assert DEFAULT_ADVISOR_MODEL == 'gemini-3.1-pro-preview'
  assert 'advisor' in TOOL_DESCRIPTION.lower()
  assert '`model_consult`' in EXECUTOR_INSTRUCTION
  assert 'senior technical advisor' in ADVISOR_SYSTEM_INSTRUCTION


def test_declaration_shape():
  """Verifies function declaration schema and required question field."""
  tool = ModelConsultTool(model=_FakeAdvisorLlm())

  decl = tool._get_declaration()

  assert decl.name == 'model_consult'
  assert decl.parameters is not None
  assert decl.parameters.required == ['question']
  assert set(decl.parameters.properties or {}) == {'question', 'context'}
  assert 'stuck' in (decl.description or '').lower()


def test_description_and_name_are_overridable():
  """Verifies custom name and description override defaults on declaration."""
  tool = ModelConsultTool(
      model=_FakeAdvisorLlm(),
      name='consult_expert',
      description='Custom escalation description.',
  )

  assert tool.name == 'consult_expert'
  assert tool._get_declaration().description == 'Custom escalation description.'


@pytest.mark.asyncio
async def test_process_llm_request_appends_executor_instruction_once():
  """Verifies process_llm_request injects EXECUTOR_INSTRUCTION without dupes."""
  tool = ModelConsultTool(model=_FakeAdvisorLlm())
  ctx = _make_tool_context([_user_event('go')])
  llm_request = LlmRequest()
  llm_request.append_instructions(['You are an SRE assistant.'])

  await tool.process_llm_request(tool_context=ctx, llm_request=llm_request)
  await tool.process_llm_request(tool_context=ctx, llm_request=llm_request)

  assert 'model_consult' in llm_request.tools_dict
  sys_inst = llm_request.config.system_instruction or ''
  assert sys_inst.count(EXECUTOR_INSTRUCTION) == 1

  renamed_tool = ModelConsultTool(model=_FakeAdvisorLlm(), name='consult_sre')
  renamed_request = LlmRequest()
  await renamed_tool.process_llm_request(
      tool_context=ctx, llm_request=renamed_request
  )
  renamed_inst = renamed_request.config.system_instruction or ''
  assert '`consult_sre`' in renamed_inst
  assert '`model_consult`' not in renamed_inst

  custom_tool = ModelConsultTool(
      model=_FakeAdvisorLlm(),
      executor_instruction='Custom escalation rule.',
  )
  custom_request = LlmRequest()
  await custom_tool.process_llm_request(
      tool_context=ctx, llm_request=custom_request
  )
  assert (
      custom_request.config.system_instruction or ''
  ) == 'Custom escalation rule.'

  disabled_tool = ModelConsultTool(
      model=_FakeAdvisorLlm(),
      executor_instruction='',
  )
  disabled_request = LlmRequest()
  await disabled_tool.process_llm_request(
      tool_context=ctx, llm_request=disabled_request
  )
  assert not (disabled_request.config.system_instruction or '')


@pytest.mark.asyncio
async def test_returns_guidance_and_accounting():
  """Verifies successful advisor consult returns guidance, usage, and budget."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, max_uses=2, session_max_uses=5)
  ctx = _make_tool_context([_user_event('Why is checkout slow?')])

  result = await _run(
      tool, ctx, question='Should I bisect deploys or profile CPU?'
  )

  assert result['status'] == 'ok'
  assert result['guidance'] == '1. Diagnosis. 2. Plan. 3. Watch out.'
  assert result['advisor_model'] == 'fake-advisor-001'
  assert result['thinking_level'] == 'high'
  assert result['consults'] == {
      'used_this_turn': 1,
      'max_uses': 2,
      'used_this_session': 1,
      'session_max_uses': 5,
      'remaining': 1,
  }
  assert result['usage']['prompt_tokens'] == 1000
  assert result['usage']['thoughts_tokens'] == 50
  assert result['latency_ms'] >= 0


@pytest.mark.asyncio
async def test_advisor_sees_session_and_question():
  """Verifies session tool calls, tool results, and handoff reach advisor."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm)
  ctx = _make_tool_context([
      _user_event('Investigate the paging alert.'),
      _agent_event([
          types.Part(
              function_call=types.FunctionCall(
                  id='fc-1', name='query_logs', args={'service': 'checkout'}
              )
          )
      ]),
      _tool_result_event('query_logs', {'errors': 42}),
  ])

  await _run(
      tool,
      ctx,
      question='Which subsystem should I inspect next?',
      context='p99 latency is flat across regions',
  )

  request = llm.requests[0]
  texts = _extract_texts(request.contents)
  assert 'Investigate the paging alert.' in texts
  assert any('[tool_call] query_logs' in text for text in texts)
  assert any(
      '[tool_result] query_logs -> {"errors": 42}' in text for text in texts
  )

  handoff = request.contents[-1].parts[-1].text or ''
  assert handoff.startswith('--- END OF EXECUTOR SESSION ---')
  assert request.contents[-1].role == 'user'
  assert ' (executor)' in handoff
  assert 'Which subsystem should I inspect next?' in handoff
  assert 'p99 latency is flat across regions' in handoff


def _extract_texts(contents: list[types.Content]) -> list[str]:
  """Extracts all non-empty text strings from a list of Content messages."""
  texts: list[str] = []
  for content in contents:
    for part in content.parts or []:
      if part.text:
        texts.append(part.text)
  return texts


@pytest.mark.asyncio
async def test_contents_never_repeat_a_role():
  """Verifies adjacent turns in advisor contents strictly alternate roles."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm)
  ctx = _make_tool_context([
      _user_event('first'),
      _agent_event([types.Part(text='reply')]),
      _tool_result_event('query_logs', {'errors': 1}),
  ])

  await _run(tool, ctx, question='Next?')

  roles = [content.role for content in llm.requests[0].contents]
  assert all(left != right for left, right in zip(roles, roles[1:]))


@pytest.mark.asyncio
async def test_executor_instruction_is_forwarded_to_advisor():
  """Verifies executor instruction reaches advisor without escalation rules."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, include_agent_instruction=True)
  ctx = _make_tool_context(
      [_user_event('go')],
      instruction=(
          f'Never restart production databases.\n\n{EXECUTOR_INSTRUCTION}'
      ),
  )

  await _run(tool, ctx, question='Can I restart the DB?')

  system_inst = llm.requests[0].config.system_instruction
  assert isinstance(system_inst, str)
  assert 'senior technical advisor' in system_inst
  assert 'Never restart production databases.' in system_inst
  assert EXECUTOR_INSTRUCTION not in system_inst


@pytest.mark.asyncio
async def test_executor_instruction_withheld_when_disabled():
  """Verifies executor agent instruction is omitted when disabled."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, include_agent_instruction=False)
  ctx = _make_tool_context(
      [_user_event('go')], instruction='Never restart production databases.'
  )

  await _run(tool, ctx, question='Can I restart the DB?')

  assert (
      'Never restart production databases.'
      not in llm.requests[0].config.system_instruction
  )


@pytest.mark.asyncio
async def test_executor_instruction_injects_state_and_static_instruction():
  """Verifies {state} placeholders and static_instruction reach the advisor."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm)
  session = Session(
      id='session-1',
      app_name='app',
      user_id='user-1',
      state={'target_env': 'prod-eu-west'},
      events=[_user_event('go')],
  )
  ctx = _make_tool_context(
      session=session,
      instruction='Only inspect cluster {target_env}.',
      static_instruction=types.Content(
          role='user',
          parts=[types.Part(text='Global policy: read-only mode.')],
      ),
  )

  await _run(tool, ctx, question='Which cluster?')

  system_inst = llm.requests[0].config.system_instruction
  assert 'Global policy: read-only mode.' in system_inst
  assert 'Only inspect cluster prod-eu-west.' in system_inst

  # Verify string static_instruction and fallback when an unset {placeholder}
  # coexists with a populated {target_env} state key.
  ctx_fallback = _make_tool_context(
      session=session,
      instruction='Cluster {target_env} with {unset_var}.',
      static_instruction='String static instruction.',
      invocation_id='inv-2',
  )
  await _run(tool, ctx_fallback, question='Fallback check?')
  system_inst_2 = llm.requests[1].config.system_instruction
  assert 'String static instruction.' in system_inst_2
  assert 'Cluster prod-eu-west with {unset_var}.' in system_inst_2

  # Verify callable instruction provider (bypass_state_injection=True)
  ctx_provider = _make_tool_context(
      session=session,
      instruction=lambda _: 'Callable provider {target_env} literal.',
      invocation_id='inv-3',
  )
  await _run(tool, ctx_provider, question='Provider check?')
  system_inst_3 = llm.requests[2].config.system_instruction
  assert 'Callable provider {target_env} literal.' in system_inst_3

  # Verify Part and list ContentUnion forms of static_instruction.
  ctx_part = _make_tool_context(
      session=session,
      instruction='Dynamic instruction.',
      static_instruction=types.Part(text='Part static instruction.'),
      invocation_id='inv-4',
  )
  await _run(tool, ctx_part, question='Part static check?')
  system_inst_4 = llm.requests[3].config.system_instruction
  assert 'Part static instruction.' in system_inst_4

  ctx_list = _make_tool_context(
      session=session,
      instruction='Dynamic instruction.',
      static_instruction=[
          'List static part 1.',
          types.Part(text='List static part 2.'),
          {'text': 'Dict static part 3.'},
          types.Part.from_bytes(data=b'img', mime_type='image/png'),
          types.File(uri='gs://bucket/doc.pdf'),
      ],
      invocation_id='inv-5',
  )
  await _run(tool, ctx_list, question='List static check?')
  system_inst_5 = llm.requests[4].config.system_instruction
  assert (
      'List static part 1.\nList static part 2.\nDict static part 3.'
      in system_inst_5
  )


@pytest.mark.asyncio
async def test_pending_model_consult_call_is_not_duplicated():
  """Verifies in-flight model_consult calls are skipped while completed stay."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm)
  ctx = _make_tool_context(
      [
          _user_event('go'),
          Event(
              author='executor',
              content=types.Content(role='model', parts=[]),
          ),
          _agent_event([
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-answered',
                      name='model_consult',
                      args={'question': 'Earlier question?'},
                  )
              )
          ]),
          Event(
              author='executor',
              content=types.Content(
                  role='user',
                  parts=[
                      types.Part(
                          function_response=types.FunctionResponse(
                              id='fc-answered',
                              name='model_consult',
                              response={'guidance': 'Check connection pool.'},
                          )
                      )
                  ],
              ),
          ),
          _agent_event([
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-current',
                      name='model_consult',
                      args={'question': 'What now?'},
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-sibling-parallel',
                      name='model_consult',
                      args={'question': 'Parallel question?'},
                  )
              ),
          ]),
      ],
      function_call_id='fc-current',
  )

  await _run(tool, ctx, question='What now?')

  texts = _extract_texts(llm.requests[0].contents)
  assert any('Earlier question?' in text for text in texts)
  assert any('Check connection pool.' in text for text in texts)
  assert not any('Parallel question?' in text for text in texts)
  assert not any(
      '[tool_call] model_consult' in text and 'What now?' in text
      for text in texts
  )


@pytest.mark.asyncio
async def test_transcript_mode_folds_session_into_one_turn():
  """Verifies transcript mode collapses session into a single user Content."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(
      model=llm,
      context_config=ModelConsultContextConfig(mode='transcript'),
  )
  ctx = _make_tool_context([
      _user_event('go'),
      _agent_event([types.Part(text='checking logs')]),
  ])

  await _run(tool, ctx, question='Next?')

  contents = llm.requests[0].contents
  assert len(contents) == 1
  assert contents[0].role == 'user'
  assert len(contents[0].parts) == 2
  assert 'EXECUTOR SESSION TRANSCRIPT' in (contents[0].parts[0].text or '')
  assert 'USER: go' in (contents[0].parts[0].text or '')
  assert 'AGENT: checking logs' in (contents[0].parts[0].text or '')
  assert (contents[0].parts[-1].text or '').startswith(
      '--- END OF EXECUTOR SESSION ---'
  )


@pytest.mark.asyncio
async def test_transcript_mode_does_not_charge_media_bytes_against_max_chars():
  """Verifies transcript mode converts media to text before char budgeting."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(
      model=llm,
      context_config=ModelConsultContextConfig(
          mode='transcript', max_chars=500, include_media=True
      ),
  )
  ctx = _make_tool_context([
      _user_event('Initial root cause clue'),
      _agent_event([types.Part(text='Middle investigation note')]),
      Event(
          invocation_id='inv-1',
          author='user',
          content=types.Content(
              role='user',
              parts=[
                  types.Part(text='Screenshot attached'),
                  types.Part(
                      inline_data=types.Blob(
                          mime_type='image/png', data=b'x' * 10_000
                      )
                  ),
              ],
          ),
      ),
  ])

  await _run(tool, ctx, question='Next?')

  transcript_part = llm.requests[0].contents[0].parts[0].text or ''
  assert 'Middle investigation note' in transcript_part
  assert '[media: image/png' in transcript_part


@pytest.mark.parametrize(
    'level,expected_enum,expected_name',
    [
        ('minimal', types.ThinkingLevel.MINIMAL, 'minimal'),
        ('low', types.ThinkingLevel.LOW, 'low'),
        ('medium', types.ThinkingLevel.MEDIUM, 'medium'),
        ('high', types.ThinkingLevel.HIGH, 'high'),
        (types.ThinkingLevel.HIGH, types.ThinkingLevel.HIGH, 'high'),
    ],
)
@pytest.mark.asyncio
async def test_thinking_level_reaches_request(
    level: str | types.ThinkingLevel,
    expected_enum: types.ThinkingLevel,
    expected_name: str,
):
  """Verifies string and enum thinking levels populate ThinkingConfig."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, thinking_level=level)
  ctx = _make_tool_context([_user_event('go')])

  result = await _run(tool, ctx, question='Next?')

  assert llm.requests[0].config.thinking_config.thinking_level == expected_enum
  assert result['thinking_level'] == expected_name


@pytest.mark.asyncio
async def test_thinking_level_none_sends_no_thinking_config():
  """Verifies thinking_level=None omits ThinkingConfig from advisor request."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, thinking_level=None)
  ctx = _make_tool_context([_user_event('go')])

  result = await _run(tool, ctx, question='Next?')

  assert llm.requests[0].config.thinking_config is None
  assert result['thinking_level'] is None


@pytest.mark.parametrize(
    'kwargs,error_match',
    [
        ({'thinking_level': 'turbo'}, 'thinking_level'),
        ({'max_uses': 0}, 'max_uses'),
        ({'max_uses': -1}, 'max_uses'),
        ({'session_max_uses': 0}, 'session_max_uses'),
        ({'session_max_uses': -2}, 'session_max_uses'),
        ({'max_output_tokens': 0}, 'max_output_tokens'),
        (
            {
                'generate_content_config': types.GenerateContentConfig(
                    max_output_tokens=0
                )
            },
            'generate_content_config.max_output_tokens',
        ),
        (
            {
                'max_output_tokens': 2048,
                'generate_content_config': types.GenerateContentConfig(
                    max_output_tokens=512
                ),
            },
            'Conflicting max_output_tokens',
        ),
        ({'timeout_seconds': 0}, 'timeout_seconds'),
        ({'model': '   '}, 'non-empty model string'),
    ],
)
def test_invalid_init_arguments_rejected_at_construction(
    kwargs: dict[str, Any], error_match: str
):
  """Verifies invalid init parameters raise ValueError at construction."""
  init_kwargs: dict[str, Any] = {'model': _FakeAdvisorLlm(), **kwargs}
  with pytest.raises(ValueError, match=error_match):
    ModelConsultTool(**init_kwargs)


def test_model_string_resolves_through_adk_registry():
  """Verifies model string resolves to a BaseLlm via LLMRegistry."""
  tool = ModelConsultTool(model='gemini-3.1-pro-preview')

  assert tool.advisor_model.model == 'gemini-3.1-pro-preview'
  assert type(tool.advisor_model).__name__ == 'Gemini'


@pytest.mark.asyncio
async def test_multiple_tool_instances_have_independent_budgets():
  """Verifies distinct ModelConsultTool names track separate use budgets."""
  llm = _FakeAdvisorLlm()
  arch_tool = ModelConsultTool(
      model=llm, name='consult_arch', max_uses=1, session_max_uses=1
  )
  sec_tool = ModelConsultTool(
      model=llm, name='consult_sec', max_uses=1, session_max_uses=1
  )
  session = Session(
      id='session-1', app_name='app', user_id='user-1', state={}, events=[]
  )
  ctx = _make_tool_context(
      [_user_event('review design')], session=session, invocation_id='inv-1'
  )

  r_arch_1 = await _run(arch_tool, ctx, question='Check architecture')
  r_arch_2 = await _run(arch_tool, ctx, question='Check architecture again')
  r_sec_1 = await _run(sec_tool, ctx, question='Check security')

  assert r_arch_1['status'] == 'ok'
  assert r_arch_2['status'] == 'limit_reached'
  assert r_sec_1['status'] == 'ok'


@pytest.mark.asyncio
async def test_generate_content_config_does_not_mutate_input():
  """Verifies caller config is not mutated and max_output_tokens syncs."""
  llm = _FakeAdvisorLlm()
  caller_cfg = types.GenerateContentConfig(temperature=0.2)
  tool = ModelConsultTool(
      model=llm,
      max_output_tokens=2048,
      generate_content_config=caller_cfg,
  )
  ctx = _make_tool_context([_user_event('go')])

  await _run(tool, ctx, question='Next?')

  sent_cfg = llm.requests[0].config
  assert sent_cfg.temperature == 0.2
  assert sent_cfg.max_output_tokens == 2048
  assert tool.max_output_tokens == 2048
  assert caller_cfg.max_output_tokens is None
  assert sent_cfg.system_instruction

  cfg_with_tokens = types.GenerateContentConfig(
      temperature=0.3, max_output_tokens=512
  )
  tool_from_cfg = ModelConsultTool(
      model=llm,
      generate_content_config=cfg_with_tokens,
  )
  assert tool_from_cfg.max_output_tokens == 512
  await _run(tool_from_cfg, ctx, question='Second?')
  assert llm.requests[1].config.max_output_tokens == 512


@pytest.mark.asyncio
async def test_max_uses_enforced_per_turn_and_resets_next_turn():
  """Verifies turn max_uses blocks excess calls and resets on next turn."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, max_uses=1)
  session = Session(
      id='session-1', app_name='app', user_id='user-1', state={}, events=[]
  )

  turn1 = _make_tool_context(
      [_user_event('turn 1')], session=session, invocation_id='inv-1'
  )
  assert tool.has_remaining_budget(turn1) is True
  first = await _run(tool, turn1, question='q1')
  assert tool.has_remaining_budget(turn1) is False
  second = await _run(tool, turn1, question='q2')

  assert first['status'] == 'ok'
  assert second['status'] == 'limit_reached'
  assert 'for this turn is exhausted (1 of 1 used)' in second['message']
  assert second['consults'] == {
      'used_this_turn': 1,
      'max_uses': 1,
      'used_this_session': 1,
      'session_max_uses': None,
      'remaining': 0,
  }
  assert len(llm.requests) == 1

  turn2 = _make_tool_context(
      [_user_event('turn 2')], session=session, invocation_id='inv-2'
  )
  assert tool.has_remaining_budget(turn2) is True
  third = await _run(tool, turn2, question='q3')
  assert third['status'] == 'ok'
  assert third['consults']['used_this_turn'] == 1
  assert third['consults']['used_this_session'] == 2
  assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_session_max_uses_enforced_across_turns():
  """Verifies session_max_uses persists across turns and blocks once reached."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, max_uses=2, session_max_uses=2)
  session = Session(
      id='session-1', app_name='app', user_id='user-1', state={}, events=[]
  )

  turn1 = _make_tool_context(
      [_user_event('turn 1')], session=session, invocation_id='inv-1'
  )
  r1 = await _run(tool, turn1, question='q1')
  assert r1['status'] == 'ok'
  assert r1['consults']['remaining'] == 1

  turn2 = _make_tool_context(
      [_user_event('turn 2')], session=session, invocation_id='inv-2'
  )
  r2 = await _run(tool, turn2, question='q2')
  assert r2['status'] == 'ok'
  assert r2['consults']['remaining'] == 0

  # Third turn has a fresh turn budget (0/2), but session budget (2/2) is full.
  turn3 = _make_tool_context(
      [_user_event('turn 3')], session=session, invocation_id='inv-3'
  )
  assert tool.has_remaining_budget(turn3) is False
  r3 = await _run(tool, turn3, question='q3')
  assert r3['status'] == 'limit_reached'
  assert 'for this session is exhausted (2 of 2 used)' in r3['message']
  assert r3['consults'] == {
      'used_this_turn': 0,
      'max_uses': 2,
      'used_this_session': 2,
      'session_max_uses': 2,
      'remaining': 0,
  }
  assert len(llm.requests) == 2
  assert session.state['model_consult:model_consult:session_uses'] == 2


@pytest.mark.asyncio
async def test_session_max_uses_without_turn_cap_and_standalone_token_cap():
  """Verifies session_max_uses when max_uses is None and standalone cap."""
  llm = _FakeAdvisorLlm(responses=[_text_response(model_version=None)])
  tool = ModelConsultTool(
      model=llm,
      max_uses=None,
      session_max_uses=2,
      max_output_tokens=1024,
  )
  ctx = _make_tool_context([_user_event('turn 1')])

  r1 = await _run(tool, ctx, question='q1')

  assert r1['status'] == 'ok'
  assert r1['advisor_model'] == 'fake-advisor'
  assert llm.requests[0].config.max_output_tokens == 1024
  assert r1['consults'] == {
      'used_this_turn': 1,
      'max_uses': None,
      'used_this_session': 1,
      'session_max_uses': 2,
      'remaining': 1,
  }


@pytest.mark.asyncio
async def test_parallel_consult_calls_respect_caps_and_preserve_deltas():
  """Verifies parallel model_consult calls serialize budgets and state_delta."""
  # 1) Turn cap saturation only (max_uses=1, session_max_uses=5).
  llm_turn_cap = _FakeAdvisorLlm(delay_seconds=0.02)
  tool_turn_cap = ModelConsultTool(
      model=llm_turn_cap, max_uses=1, session_max_uses=5
  )
  session_turn = Session(
      id='s-turn', app_name='app', user_id='u1', state={}, events=[]
  )
  ctx_turn_a = _make_tool_context(
      [_user_event('go')],
      session=session_turn,
      invocation_id='inv-turn',
      function_call_id='fc-a',
  )
  ctx_turn_b = _make_tool_context(
      [_user_event('go')],
      session=session_turn,
      invocation_id='inv-turn',
      function_call_id='fc-b',
  )
  res_ta, res_tb = await asyncio.gather(
      _run(tool_turn_cap, ctx_turn_a, question='q1'),
      _run(tool_turn_cap, ctx_turn_b, question='q2'),
  )
  assert sorted([res_ta['status'], res_tb['status']]) == ['limit_reached', 'ok']
  assert len(llm_turn_cap.requests) == 1

  # 2) Session cap saturation only (max_uses=5, session_max_uses=1).
  llm_sess_cap = _FakeAdvisorLlm(delay_seconds=0.02)
  tool_sess_cap = ModelConsultTool(
      model=llm_sess_cap, max_uses=5, session_max_uses=1
  )
  session_sess = Session(
      id='s-sess', app_name='app', user_id='u1', state={}, events=[]
  )
  ctx_sess_a = _make_tool_context(
      [_user_event('go')],
      session=session_sess,
      invocation_id='inv-sess',
      function_call_id='fc-sa',
  )
  ctx_sess_b = _make_tool_context(
      [_user_event('go')],
      session=session_sess,
      invocation_id='inv-sess',
      function_call_id='fc-sb',
  )
  res_sa, res_sb = await asyncio.gather(
      _run(tool_sess_cap, ctx_sess_a, question='q1'),
      _run(tool_sess_cap, ctx_sess_b, question='q2'),
  )
  assert sorted([res_sa['status'], res_sb['status']]) == ['limit_reached', 'ok']
  assert len(llm_sess_cap.requests) == 1

  # Now test max_uses=5 where Call 1 takes longer than Call 2 so Call 2 finishes
  # first, and verify merge_parallel_function_response_events preserves count=2.
  llm_cap5 = _FakeAdvisorLlm(per_call_delays=[0.03, 0.005, 0.03, 0.005])
  tool_cap5 = ModelConsultTool(model=llm_cap5, max_uses=5, session_max_uses=5)
  session_service = InMemorySessionService()
  session_cap5 = await session_service.create_session(
      app_name='app', user_id='u1', session_id='s5'
  )
  inv_ctx = InvocationContext(
      session_service=session_service,
      invocation_id='inv-5',
      agent=LlmAgent(name='executor', model='gemini-2.5-flash'),
      session=session_cap5,
  )
  ctx5_1 = ToolContext(
      inv_ctx, function_call_id='fc-1', event_actions=EventActions()
  )
  ctx5_2 = ToolContext(
      inv_ctx, function_call_id='fc-2', event_actions=EventActions()
  )

  r5_1, r5_2 = await asyncio.gather(
      _run(tool_cap5, ctx5_1, question='q1'),
      _run(tool_cap5, ctx5_2, question='q2'),
  )
  ev1 = Event(
      invocation_id='inv-5',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(
                  name='model_consult', response=r5_1
              )
          ],
      ),
      actions=ctx5_1.actions,
  )
  ev2 = Event(
      invocation_id='inv-5',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(
                  name='model_consult', response=r5_2
              )
          ],
      ),
      actions=ctx5_2.actions,
  )
  merged_event = merge_parallel_function_response_events([ev1, ev2])
  await session_service.append_event(session=session_cap5, event=merged_event)

  assert session_cap5.state['model_consult:model_consult:session_uses'] == 2
  assert session_cap5.state['temp:model_consult:model_consult:inv-5:uses'] == 2

  # Also verify reverse completion order (when fc-2 finishes before fc-1) still
  # merges state_delta to 2 rather than overwriting 2 back to 1.
  session_rev = await session_service.create_session(
      app_name='app', user_id='u1'
  )
  ctx_rev_1 = _make_tool_context(
      [_user_event('rev')],
      session=session_rev,
      invocation_id='inv-rev',
      function_call_id='fc-rev-1',
  )
  ctx_rev_2 = _make_tool_context(
      [_user_event('rev')],
      session=session_rev,
      invocation_id='inv-rev',
      function_call_id='fc-rev-2',
  )

  r_rev_1, r_rev_2 = await asyncio.gather(
      _run(tool_cap5, ctx_rev_1, question='rev-1'),
      _run(tool_cap5, ctx_rev_2, question='rev-2'),
  )
  ev_rev_1 = Event(
      invocation_id='inv-rev',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(
                  name='model_consult', response=r_rev_1
              )
          ],
      ),
      actions=ctx_rev_1.actions,
  )
  ev_rev_2 = Event(
      invocation_id='inv-rev',
      author='executor',
      content=types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(
                  name='model_consult', response=r_rev_2
              )
          ],
      ),
      actions=ctx_rev_2.actions,
  )
  merged_rev = merge_parallel_function_response_events([ev_rev_1, ev_rev_2])
  await session_service.append_event(session=session_rev, event=merged_rev)
  assert session_rev.state['model_consult:model_consult:session_uses'] == 2
  assert session_rev.state['temp:model_consult:model_consult:inv-rev:uses'] == 2

  # Verify two sequential consults in the same invocation do not mutate the
  # already-emitted first event's state_delta (inv_deltas is pruned when
  # active_calls drops to 0).
  session_seq = await session_service.create_session(
      app_name='app', user_id='u1'
  )
  ctx_seq_1 = _make_tool_context(
      [_user_event('seq')],
      session=session_seq,
      invocation_id='inv-seq',
      function_call_id='fc-seq-1',
  )
  ctx_seq_2 = _make_tool_context(
      [_user_event('seq')],
      session=session_seq,
      invocation_id='inv-seq',
      function_call_id='fc-seq-2',
  )
  await _run(tool_cap5, ctx_seq_1, question='seq-1')
  assert (
      ctx_seq_1.actions.state_delta[
          'temp:model_consult:model_consult:inv-seq:uses'
      ]
      == 1
  )
  await _run(tool_cap5, ctx_seq_2, question='seq-2')
  assert (
      ctx_seq_1.actions.state_delta[
          'temp:model_consult:model_consult:inv-seq:uses'
      ]
      == 1
  )
  assert (
      ctx_seq_2.actions.state_delta[
          'temp:model_consult:model_consult:inv-seq:uses'
      ]
      == 2
  )


@pytest.mark.asyncio
async def test_session_max_uses_persists_with_strict_state_schema():
  """Verifies session_max_uses works even when State enforces a state_schema."""

  class _StrictSchema(BaseModel):
    allowed_field: str = 'ok'

  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, session_max_uses=1)
  session = Session(
      id='session-strict', app_name='app', user_id='u1', state={}, events=[]
  )
  ctx1 = _make_tool_context(
      [_user_event('t1')], session=session, invocation_id='inv-1'
  )
  ctx1._state = State(
      value=session.state,
      delta=ctx1.actions.state_delta,
      schema=_StrictSchema,
  )

  r1 = await _run(tool, ctx1, question='q1')
  assert r1['status'] == 'ok'
  assert session.state['model_consult:model_consult:session_uses'] == 1
  assert (
      ctx1.actions.state_delta['model_consult:model_consult:session_uses'] == 1
  )

  ctx2 = _make_tool_context(
      [_user_event('t2')], session=session, invocation_id='inv-2'
  )
  ctx2._state = State(
      value=session.state,
      delta=ctx2.actions.state_delta,
      schema=_StrictSchema,
  )
  r2 = await _run(tool, ctx2, question='q2')
  assert r2['status'] == 'limit_reached'
  assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_missing_question_rejected_without_calling_advisor():
  """Verifies blank or missing question returns invalid_request immediately."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm)
  ctx = _make_tool_context([_user_event('go')])

  result_blank = await _run(tool, ctx, question='   ')
  result_missing = await tool.run_async(args={}, tool_context=ctx)

  assert result_blank['status'] == 'invalid_request'
  assert result_missing['status'] == 'invalid_request'
  assert llm.requests == []


@pytest.mark.asyncio
async def test_advisor_failure_degrades_gracefully_without_burning_budget():
  """Verifies advisor runtime error returns status='error' and keeps budget."""
  failing_llm = _FakeAdvisorLlm(
      errors=[RuntimeError('503 backend unavailable')]
  )
  tool = ModelConsultTool(model=failing_llm, max_uses=1, session_max_uses=1)
  ctx = _make_tool_context([_user_event('go')])

  result = await _run(tool, ctx, question='Next?')

  assert result['status'] == 'error'
  assert '503' in result['error']
  assert 'own best judgment' in result['message']
  assert result['consults']['used_this_turn'] == 0
  assert result['consults']['used_this_session'] == 0
  assert result['consults']['remaining'] == 1


@pytest.mark.asyncio
async def test_advisor_timeout_degrades_gracefully_without_burning_budget():
  """Verifies advisor timeout returns status='error' and keeps budget."""
  slow_llm = _FakeAdvisorLlm(delay_seconds=0.2)
  timeout_tool = ModelConsultTool(
      model=slow_llm, max_uses=1, session_max_uses=1, timeout_seconds=0.01
  )
  ctx = _make_tool_context([_user_event('go')])

  timeout_result = await _run(timeout_tool, ctx, question='Next?')

  assert timeout_result['status'] == 'error'
  assert 'timed out' in timeout_result['error']
  assert timeout_result['consults']['used_this_turn'] == 0
  assert timeout_result['consults']['remaining'] == 1


@pytest.mark.asyncio
async def test_thinking_config_rejection_falls_back_and_still_answers():
  """Verifies unsupported thinking_level falls back without thinking_config."""
  llm = _FakeAdvisorLlm(
      responses=[_text_response('fallback advice')],
      errors=[
          ValueError('thinking_level is not supported by this model'),
          None,
      ],
  )
  tool = ModelConsultTool(model=llm, thinking_level='high')
  ctx = _make_tool_context([_user_event('go')])

  result = await _run(tool, ctx, question='Next?')

  assert result['status'] == 'ok'
  assert result['guidance'] == 'fallback advice'
  assert len(llm.requests) == 2
  assert llm.requests[0].config.thinking_config is not None
  assert llm.requests[1].config.thinking_config is None


@pytest.mark.asyncio
async def test_advisor_receives_executor_tool_inventory():
  """Verifies executor tools and truncated descriptions reach advisor prompt."""

  def list_deploys(service: str) -> dict[str, str]:
    """Lists recent deploys for a service."""
    return {'service': service}

  def verbose_tool(query: str) -> str:
    return query

  verbose_tool.__doc__ = 'A' * 350

  def no_doc_tool(x: str) -> str:
    return x

  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(
      model=llm,
      advisor_instruction='Custom advisor system prompt.',
      max_uses=2,
  )
  ctx = _make_tool_context(
      [_user_event('go')],
      tools=[list_deploys, verbose_tool, no_doc_tool, tool],
  )
  assert ctx._invocation_context.canonical_tools_cache is None

  await _run(tool, ctx, question='What next?')

  assert ctx._invocation_context.canonical_tools_cache is not None
  system = llm.requests[0].config.system_instruction
  assert isinstance(system, str)
  assert system.startswith('Custom advisor system prompt.')
  assert 'TOOLS AVAILABLE TO THE EXECUTOR' in system
  inventory_section = system.split('TOOLS AVAILABLE TO THE EXECUTOR')[1]
  assert (
      '- list_deploys: Lists recent deploys for a service.' in inventory_section
  )
  assert f"- verbose_tool: {'A' * 300}..." in inventory_section
  assert '- no_doc_tool' in inventory_section
  assert '- no_doc_tool:' not in inventory_section
  assert 'model_consult' not in inventory_section

  # Second call within the same invocation reuses canonical_tools_cache
  # without calling agent.canonical_tools again.
  async def _fail_if_called(_):
    raise AssertionError('canonical_tools should not be re-resolved')

  object.__setattr__(
      ctx._invocation_context.agent, 'canonical_tools', _fail_if_called
  )
  await _run(tool, ctx, question='Second check?')
  system_2 = llm.requests[1].config.system_instruction
  assert '- list_deploys: Lists recent deploys for a service.' in system_2


@pytest.mark.asyncio
async def test_tool_inventory_withheld_when_disabled():
  """Verifies include_tool_inventory=False omits tool list from prompt."""

  def list_deploys(service: str) -> dict[str, str]:
    """Lists recent deploys for a service."""
    return {'service': service}

  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, include_tool_inventory=False)
  ctx = _make_tool_context([_user_event('go')], tools=[list_deploys, tool])

  await _run(tool, ctx, question='What next?')

  assert (
      'TOOLS AVAILABLE TO THE EXECUTOR'
      not in llm.requests[0].config.system_instruction
  )


@pytest.mark.asyncio
async def test_corrupt_state_and_broken_agent_callbacks_degrade_gracefully(
    caplog: pytest.LogCaptureFixture,
):
  """Verifies corrupt state counters and broken callbacks do not crash."""
  llm = _FakeAdvisorLlm()
  tool = ModelConsultTool(model=llm, max_uses=3)
  ctx = _make_tool_context(
      [_user_event('go')],
      instruction='Executor rule.',
      invocation_id='',
  )
  ctx.state[tool._turn_uses_state_key(ctx)] = -5
  ctx.state[tool._session_uses_state_key()] = 'not-an-int'
  object.__setattr__(ctx._invocation_context.agent, 'name', 123)

  res0 = await _run(tool, ctx, question='Non-str agent name check?')
  assert res0['status'] == 'ok'
  assert '(the executor)' in llm.requests[0].config.system_instruction
  assert tool._turn_uses_state_key(ctx).endswith(':unknown:uses')

  ctx._invocation_context.agent.name = 'unknown'

  async def _broken_instruction(_):
    raise RuntimeError('instruction callback boom')

  async def _broken_tools(_):
    raise RuntimeError('tools callback boom')

  ctx._invocation_context.canonical_tools_cache = None
  object.__setattr__(
      ctx._invocation_context.agent,
      'canonical_instruction',
      _broken_instruction,
  )
  object.__setattr__(
      ctx._invocation_context.agent,
      'canonical_tools',
      _broken_tools,
  )

  res = await _run(tool, ctx, question=12345, context=67890)

  assert res['status'] == 'ok'
  assert res['consults']['used_this_turn'] == 2
  assert res['consults']['used_this_session'] == 2
  handoff_text = llm.requests[1].contents[-1].parts[-1].text or ''
  assert '(unknown)' not in handoff_text
  assert '12345' in handoff_text
  assert '67890' in handoff_text

  class _RaisingStateDict(dict):
    """State mapping that raises RuntimeError on write."""

    def __setitem__(self, key, value):
      raise RuntimeError('storage write failure')

  # Verify non-callable instruction/tools attributes and failing state write.
  ctx._invocation_context.canonical_tools_cache = None
  object.__setattr__(ctx._invocation_context.agent, 'canonical_instruction', 42)
  object.__setattr__(ctx._invocation_context.agent, 'canonical_tools', 42)
  object.__setattr__(ctx, '_state', _RaisingStateDict())
  caplog.clear()
  res2 = await _run(tool, ctx, question='Still works?')
  assert res2['status'] == 'ok'
  assert any(
      record.levelname == 'WARNING'
      and 'ModelConsultTool could not persist its use counters'
      in record.getMessage()
      for record in caplog.records
  )
