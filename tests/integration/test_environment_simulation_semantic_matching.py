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

"""Integration test: an agent replays a recorded tool call semantically.

A real model drives the agent, and the same real model judges whether the
agent's tool arguments mean the same thing as the recorded ones. The recorded
injection uses full names ("software engineers", "New York City") while the
agent is told to pass the user's abbreviations ("SWEs", "NYC"), so the call can
only hit the injection through the semantic matcher, never by exact equality.

Requires model credentials, as the other integration tests do: GOOGLE_API_KEY
for the Google AI backend, or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
with Application Default Credentials for Vertex AI. Set ADK_TEST_MODEL to use
a model other than gemini-2.5-flash.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_factory import EnvironmentSimulationFactory
from google.adk.tools.tool_args_matcher import _semantic_arg_matcher
from google.genai import types
import pytest

_MODEL = os.environ.get('ADK_TEST_MODEL', 'gemini-2.5-flash')

_RECORDED_ARGS = {'query': 'software engineers', 'location': 'New York City'}
_RECORDED_RESPONSE = {
    'source': 'recorded',
    'employees': ['Ada Lovelace', 'Alan Turing'],
}
_LIVE_RESPONSE = {'source': 'live', 'employees': []}

_INSTRUCTION = """\
You look up employees with the search_employees tool.
Always call search_employees exactly once per request.
Pass the role and the location exactly as the user wrote them, without
expanding abbreviations or correcting spelling.
After the tool returns, list the employee names it returned."""

_PARALLEL_INSTRUCTION = """\
You look up employees with the search_employees tool.
Call search_employees once for every role and location pair the user asks
about, issuing all the calls together in a single turn.
Pass each role and location exactly as the user wrote them, without
expanding abbreviations or correcting spelling.
After the tools return, list the employee names they returned."""

_SWE_ARGS = _RECORDED_ARGS
_SWE_RESPONSE = {'source': 'recorded', 'team': 'swe', 'employees': ['Ada']}
_PM_ARGS = {'query': 'product managers', 'location': 'San Francisco'}
_PM_RESPONSE = {'source': 'recorded', 'team': 'pm', 'employees': ['Grace']}

_STRICT_SCOPE_CRITERIA = (
    'The values match only if they convey the exact same meaning and scope,'
    ' allowing only rephrasing, synonyms, abbreviations, or differences in'
    ' casing and word order. They do not match if one value is broader or'
    ' narrower than the other, or if either value adds, omits, or changes any'
    ' qualifier, constraint, or detail.'
)
_BROAD_QUERY_ARGS = {'query': 'python jobs'}
_BROAD_QUERY_RESPONSE = {
    'source': 'recorded',
    'scope': 'broad',
    'jobs': ['Backend Engineer'],
}
_NARROW_QUERY_ARGS = {'query': 'python developer jobs in NYC'}
_NARROW_QUERY_RESPONSE = {
    'source': 'recorded',
    'scope': 'narrow',
    'jobs': ['NYC Python Developer'],
}
_JOBS_LIVE_RESPONSE = {'source': 'live', 'jobs': []}
_JOBS_INSTRUCTION = """\
You look up job postings with the search_jobs tool.
Always call search_jobs exactly once per request.
Pass the search phrase after "Search for:" verbatim, without adding, removing,
or expanding words or abbreviations.
After the tool returns, list the job titles it returned."""


def _skip_without_credentials() -> None:
  """Skips the test when the selected backend has no credentials."""
  if os.environ.get('GOOGLE_GENAI_USE_ENTERPRISE') == '1':
    if not os.environ.get('GOOGLE_CLOUD_PROJECT'):
      pytest.skip('GOOGLE_CLOUD_PROJECT is not set for the Vertex backend.')
  elif not os.environ.get('GOOGLE_API_KEY'):
    pytest.skip('GOOGLE_API_KEY is not set for the Google AI backend.')


class _JudgeCallTracker:
  """Records how real semantic-judge model calls overlap in time.

  The engine gives every injection config its own SemanticArgMatcher, and one
  matcher compares all the arguments of its config. So two calls in flight on
  the same matcher show ToolArgsMatcher comparing arguments concurrently, and
  calls in flight on different matchers show the engine comparing injection
  configs concurrently.
  """

  def __init__(self) -> None:
    self.calls = 0
    self.max_in_flight = 0
    self.max_on_one_matcher = 0
    self.max_matchers_in_flight = 0
    self._active: list[int] = []

  def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wraps every judge model that semantic matchers create.

    Each SemanticArgMatcher asks LLMRegistry for its model once and reuses it,
    so the identity of the returned model stands for the matcher. Only the
    registry seen by the semantic matcher module is replaced, so the agent's
    own model is left alone.
    """
    tracker = self

    class _TrackingRegistry:

      @staticmethod
      def new_llm(model: str) -> _TrackedLlm:
        return _TrackedLlm(LLMRegistry.new_llm(model), tracker)

    monkeypatch.setattr(_semantic_arg_matcher, 'LLMRegistry', _TrackingRegistry)

  def enter(self, matcher_id: int) -> None:
    self.calls += 1
    self._active.append(matcher_id)
    self.max_in_flight = max(self.max_in_flight, len(self._active))
    self.max_on_one_matcher = max(
        self.max_on_one_matcher, self._active.count(matcher_id)
    )
    self.max_matchers_in_flight = max(
        self.max_matchers_in_flight, len(set(self._active))
    )

  def leave(self, matcher_id: int) -> None:
    self._active.remove(matcher_id)


class _TrackedLlm:
  """Forwards model calls to a real model, reporting them to a tracker."""

  def __init__(self, llm: BaseLlm, tracker: _JudgeCallTracker) -> None:
    self._llm = llm
    self._tracker = tracker

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self._tracker.enter(id(self))
    try:
      async for response in self._llm.generate_content_async(
          llm_request, stream=stream
      ):
        yield response
    finally:
      self._tracker.leave(id(self))


class _Harness:
  """An agent whose search tool is simulated with semantic matching."""

  def __init__(
      self,
      recordings: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
      instruction: str = _INSTRUCTION,
  ) -> None:
    self.live_calls: list[dict[str, Any]] = []
    if recordings is None:
      recordings = [(_RECORDED_ARGS, _RECORDED_RESPONSE)]

    def search_employees(query: str, location: str) -> dict[str, Any]:
      """Searches the employee directory.

      Args:
        query: The role to search for.
        location: The office location to search in.

      Returns:
        The matching employees.
      """
      self.live_calls.append({'query': query, 'location': location})
      return _LIVE_RESPONSE

    config = EnvironmentSimulationConfig(
        simulation_model=_MODEL,
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name='search_employees',
                injection_configs=[
                    InjectionConfig.model_validate({
                        'match_args': recorded_args,
                        'args_matcher': {
                            'default_matcher': {'type': 'semantic'},
                        },
                        'injected_response': recorded_response,
                    })
                    for recorded_args, recorded_response in recordings
                ],
            )
        ],
    )
    agent = LlmAgent(
        name='directory_agent',
        model=_MODEL,
        instruction=instruction,
        tools=[search_employees],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
    self._runner = InMemoryRunner(
        agent=agent,
        app_name='semantic_matching_test',
        plugins=[EnvironmentSimulationFactory.create_plugin(config)],
    )

  async def ask(
      self, text: str
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sends one user message.

    Returns:
      The arguments of each search_employees call the agent made, and the
      response each call received.
    """
    session = await self._runner.session_service.create_session(
        app_name='semantic_matching_test', user_id='test_user'
    )
    calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    async for event in self._runner.run_async(
        user_id='test_user',
        session_id=session.id,
        new_message=types.Content(role='user', parts=[types.Part(text=text)]),
    ):
      for call in event.get_function_calls():
        if call.name == 'search_employees':
          calls.append(dict(call.args or {}))
      for response in event.get_function_responses():
        if response.name == 'search_employees':
          responses.append(dict(response.response or {}))
    return calls, responses


async def test_rephrased_call_replays_recorded_response():
  """Abbreviated arguments still hit the recorded injection."""
  _skip_without_credentials()
  harness = _Harness()

  calls, responses = await harness.ask('Find the SWEs in NYC.')

  assert calls, 'The agent did not call search_employees.'
  # Guards against a vacuous pass: if the agent had expanded the
  # abbreviations itself, exact matching alone would have hit the injection.
  assert calls[0] != _RECORDED_ARGS, (
      'The agent passed the recorded arguments verbatim, so this run did not'
      f' exercise semantic matching: {calls[0]}'
  )
  assert responses[0] == _RECORDED_RESPONSE
  assert harness.live_calls == []


async def test_unrelated_call_reaches_the_real_tool():
  """Arguments with a different meaning fall through to the real tool."""
  _skip_without_credentials()
  harness = _Harness()

  calls, responses = await harness.ask('Find the accountants in London.')

  assert calls, 'The agent did not call search_employees.'
  assert responses[0] == _LIVE_RESPONSE
  assert harness.live_calls == calls


async def test_parallel_tool_calls_are_matched_concurrently(monkeypatch):
  """Real judge calls overlap across arguments and across injection configs.

  The agent asks about two role and location pairs, and two injections are
  recorded. For each tool call, the engine compares both injection configs at
  once, and each config's ToolArgsMatcher judges both of its arguments at once,
  so the real model calls overlap at both levels.
  """
  _skip_without_credentials()
  tracker = _JudgeCallTracker()
  tracker.install(monkeypatch)
  harness = _Harness(
      recordings=[(_SWE_ARGS, _SWE_RESPONSE), (_PM_ARGS, _PM_RESPONSE)],
      instruction=_PARALLEL_INSTRUCTION,
  )

  calls, responses = await harness.ask(
      'Find the SWEs in NYC and the PMs in SF.'
  )

  assert len(calls) == 2, f'Expected two search_employees calls: {calls}'
  # Guards against a vacuous pass: verbatim arguments skip the model.
  assert _SWE_ARGS not in calls and _PM_ARGS not in calls, (
      'The agent passed recorded arguments verbatim, so this run did not'
      f' exercise semantic matching: {calls}'
  )
  # Each call replayed its own recording, and nothing reached the real tool.
  assert sorted(r['team'] for r in responses) == ['pm', 'swe']
  assert harness.live_calls == []
  # ToolArgsMatcher: both arguments of one config were judged together.
  assert tracker.max_on_one_matcher >= 2, vars(tracker)
  # Engine: both injection configs were compared together.
  assert tracker.max_matchers_in_flight >= 2, vars(tracker)


class _JobsHarness:
  """An agent whose search_jobs tool is simulated with semantic matching."""

  def __init__(
      self,
      *,
      criteria: str | None = None,
      recordings: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
  ) -> None:
    self.live_calls: list[dict[str, Any]] = []
    if recordings is None:
      recordings = [
          (_BROAD_QUERY_ARGS, _BROAD_QUERY_RESPONSE),
          (_NARROW_QUERY_ARGS, _NARROW_QUERY_RESPONSE),
      ]

    def search_jobs(query: str) -> dict[str, Any]:
      """Searches job postings.

      Args:
        query: The job search query.

      Returns:
        The matching job postings.
      """
      self.live_calls.append({'query': query})
      return _JOBS_LIVE_RESPONSE

    query_matcher: dict[str, Any] = {'type': 'semantic'}
    if criteria is not None:
      query_matcher['criteria'] = criteria
    matcher_config = {'per_arg_matchers': {'query': query_matcher}}

    config = EnvironmentSimulationConfig(
        simulation_model=_MODEL,
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name='search_jobs',
                injection_configs=[
                    InjectionConfig.model_validate({
                        'match_args': recorded_args,
                        'args_matcher': matcher_config,
                        'injected_response': recorded_response,
                    })
                    for recorded_args, recorded_response in recordings
                ],
            )
        ],
    )
    agent = LlmAgent(
        name='jobs_agent',
        model=_MODEL,
        instruction=_JOBS_INSTRUCTION,
        tools=[search_jobs],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
    self._runner = InMemoryRunner(
        agent=agent,
        app_name='jobs_semantic_matching_test',
        plugins=[EnvironmentSimulationFactory.create_plugin(config)],
    )

  async def ask(
      self, text: str
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sends one user message and returns tool call args and responses."""
    session = await self._runner.session_service.create_session(
        app_name='jobs_semantic_matching_test', user_id='test_user'
    )
    calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    async for event in self._runner.run_async(
        user_id='test_user',
        session_id=session.id,
        new_message=types.Content(role='user', parts=[types.Part(text=text)]),
    ):
      for call in event.get_function_calls():
        if call.name == 'search_jobs':
          calls.append(dict(call.args or {}))
      for response in event.get_function_responses():
        if response.name == 'search_jobs':
          responses.append(dict(response.response or {}))
    return calls, responses


async def test_default_criteria_broad_query_shadows_later_narrow_query():
  """Without custom criteria, an earlier broad query shadows a rephrased one.

  An exact call for the narrower recording replays it, because exact matches
  are tried before any semantic comparison. A rephrased narrower call has no
  exact match, and the default criteria checks whether two values share the
  same core intent, so it matches the earlier broader recording ("python
  jobs") and first-match-wins replays the broad response.
  """
  _skip_without_credentials()
  harness = _JobsHarness()

  exact_narrow_calls, exact_narrow_responses = await harness.ask(
      'Search for: python developer jobs in NYC'
  )
  assert exact_narrow_calls == [_NARROW_QUERY_ARGS]
  assert exact_narrow_responses == [_NARROW_QUERY_RESPONSE]

  rephrased_narrow_calls, rephrased_narrow_responses = await harness.ask(
      'Search for: Python dev openings in New York City'
  )
  assert rephrased_narrow_calls, 'The agent did not call search_jobs.'
  assert rephrased_narrow_calls[0] != _NARROW_QUERY_ARGS
  assert rephrased_narrow_responses == [_BROAD_QUERY_RESPONSE]
  assert harness.live_calls == []


async def test_strict_criteria_disambiguates_broad_and_narrow_queries():
  """Strict general criteria prevents broad queries from shadowing narrow ones.

  When a broad recording ("python jobs") precedes a narrower recording
  ("python developer jobs in NYC"), a general criteria requiring identical
  scope and qualifiers lets the semantic judge reject the earlier broad
  recording for narrow calls while still matching rephrasings of each.
  """
  _skip_without_credentials()
  harness = _JobsHarness(criteria=_STRICT_SCOPE_CRITERIA)

  # 1. Exact narrow query hits the second config without asking the judge.
  exact_narrow_calls, exact_narrow_responses = await harness.ask(
      'Search for: python developer jobs in NYC'
  )
  assert exact_narrow_calls == [_NARROW_QUERY_ARGS]
  assert exact_narrow_responses == [_NARROW_QUERY_RESPONSE]

  # 2. Rephrased narrow query is judged against both configs and hits the
  # second without falling into the earlier broad config.
  rephrased_narrow_calls, rephrased_narrow_responses = await harness.ask(
      'Search for: Python dev openings in New York City'
  )
  assert rephrased_narrow_calls, 'The agent did not call search_jobs.'
  assert rephrased_narrow_calls[0] != _NARROW_QUERY_ARGS, (
      'The agent passed the narrow arguments verbatim, so this run did not'
      f' exercise semantic matching: {rephrased_narrow_calls[0]}'
  )
  assert rephrased_narrow_calls[0] != _BROAD_QUERY_ARGS
  assert rephrased_narrow_responses == [_NARROW_QUERY_RESPONSE]

  # 3. Rephrased broad query still matches the first broad config semantically.
  rephrased_broad_calls, rephrased_broad_responses = await harness.ask(
      'Search for: Python job listings'
  )
  assert rephrased_broad_calls, 'The agent did not call search_jobs.'
  assert rephrased_broad_calls[0] != _BROAD_QUERY_ARGS, (
      'The agent passed the broad arguments verbatim, so this run did not'
      f' exercise semantic matching: {rephrased_broad_calls[0]}'
  )
  assert rephrased_broad_responses == [_BROAD_QUERY_RESPONSE]
  assert harness.live_calls == []
