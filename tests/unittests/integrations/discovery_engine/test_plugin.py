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

"""Tests for GroundingPlugin."""

from __future__ import annotations

from collections.abc import Sequence
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.integrations.discovery_engine import CheckGroundingClient
from google.adk.integrations.discovery_engine import Fact
from google.adk.integrations.discovery_engine import FactChunk
from google.adk.integrations.discovery_engine import GroundingLimits
from google.adk.integrations.discovery_engine import GroundingPlugin
from google.adk.integrations.discovery_engine import GroundingResult
from google.adk.integrations.discovery_engine import GroundingStatus
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from tests.unittests import testing_utils

_METADATA = types.GroundingMetadata(
    grounding_supports=[types.GroundingSupport(grounding_chunk_indices=[0])]
)


class _RecordingClient:
  """GroundingClient fake that records the facts each check receives."""

  def __init__(
      self,
      result: GroundingResult | None = None,
      limits: GroundingLimits | None = None,
  ) -> None:
    self.limits = limits or GroundingLimits()
    self.result = result
    self.sent: list[tuple[str, list[FactChunk]]] = []
    self.closed = False

  async def check_grounding_async(
      self, *, answer_candidate: str, facts: Sequence[FactChunk]
  ) -> GroundingResult:
    self.sent.append((answer_candidate, list(facts)))
    return self.result or GroundingResult(
        status=GroundingStatus.OK, num_facts_evaluated=len(facts)
    )

  async def close(self) -> None:
    self.closed = True


async def _respond(
    plugin: GroundingPlugin,
    text: str = 'An answer.',
    partial: bool = False,
    parts: list[types.Part] | None = None,
    metadata: types.GroundingMetadata | None = None,
) -> tuple[LlmResponse, LlmResponse | None]:
  """Runs a response through the plugin; returns it and any replacement."""
  response = LlmResponse(
      content=types.Content(
          role='model', parts=parts or [types.Part(text=text)]
      ),
      partial=partial,
      grounding_metadata=metadata,
  )
  replacement = await plugin.after_model_callback(
      callback_context=mock.MagicMock(spec=CallbackContext),
      llm_response=response,
  )
  return response, replacement


# Facts


async def _sent_chunks(
    plugin: GroundingPlugin, client: _RecordingClient
) -> list[FactChunk]:
  """Checks one answer and returns the chunks the client received."""
  await _respond(plugin)
  return client.sent[-1][1]


async def test_fact_is_sent_with_its_metadata():
  """A fact passed at construction is sent with title, link and attributes."""
  client = _RecordingClient()
  plugin = GroundingPlugin(
      client=client,
      facts=[
          Fact(
              text='Water boils at 100 C.',
              title='Physics',
              link='https://example.com',
              attributes={'topic': 'heat'},
          )
      ],
  )

  chunks = await _sent_chunks(plugin, client)

  assert chunks == [
      FactChunk(
          text='Water boils at 100 C.',
          title='Physics',
          link='https://example.com',
          attributes={'topic': 'heat'},
      )
  ]


async def test_long_fact_is_split_at_sentence_ends():
  """Text over ``max_fact_chars`` is chunked at the last sentence that fits."""
  client = _RecordingClient(limits=GroundingLimits(max_fact_chars=38))
  plugin = GroundingPlugin(
      client=client,
      facts=[Fact(text='One two three. Four five six. Seven eight nine.')],
  )

  chunks = await _sent_chunks(plugin, client)

  assert [c.text for c in chunks] == [
      'One two three. Four five six.',
      'Seven eight nine.',
  ]


async def test_fact_without_boundaries_is_cut_at_the_limit():
  """Text with no whitespace is cut at exactly ``max_fact_chars``."""
  client = _RecordingClient(limits=GroundingLimits(max_fact_chars=4))
  plugin = GroundingPlugin(client=client, facts=[Fact(text='abcdefghij')])

  chunks = await _sent_chunks(plugin, client)

  assert [c.text for c in chunks] == ['abcd', 'efgh', 'ij']


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('First para.\n\nSecond para.', ['First para.', 'Second para.']),
        ('First line\nsecond line', ['First line', 'second line']),
        ('alpha beta gamma delta', ['alpha beta', 'gamma delta']),
    ],
    ids=['paragraph', 'line', 'word'],
)
async def test_long_fact_prefers_paragraph_then_line_breaks(
    text: str, expected: list[str]
):
  """Paragraph, line, then word breaks are preferred split points."""
  client = _RecordingClient(limits=GroundingLimits(max_fact_chars=14))
  plugin = GroundingPlugin(client=client, facts=[Fact(text=text)])

  chunks = await _sent_chunks(plugin, client)

  assert [c.text for c in chunks] == expected


async def test_identical_facts_are_sent_once():
  """Facts with the same content get the same ID and are sent once."""
  client = _RecordingClient()
  first = Fact(text='Leave is 16 weeks.', title='Handbook')
  second = Fact(text='Leave is 16 weeks.', title='Handbook')
  plugin = GroundingPlugin(client=client, facts=[first, second])

  chunks = await _sent_chunks(plugin, client)

  assert first.id == second.id
  assert [c.text for c in chunks] == ['Leave is 16 weeks.']


async def test_same_text_with_different_attributes_is_kept_twice():
  """Facts that differ only in attributes are separate facts."""
  client = _RecordingClient()
  plugin = GroundingPlugin(
      client=client,
      facts=[
          Fact(text='Leave is 16 weeks.', attributes={'tenant': 'a'}),
          Fact(text='Leave is 16 weeks.', attributes={'tenant': 'b'}),
      ],
  )

  chunks = await _sent_chunks(plugin, client)

  assert [c.attributes for c in chunks] == [{'tenant': 'a'}, {'tenant': 'b'}]


async def test_later_fact_with_the_same_id_replaces_the_earlier_one():
  """Of two facts with the same ID, only the later one is sent."""
  client = _RecordingClient()
  plugin = GroundingPlugin(
      client=client,
      facts=[
          Fact(text='Old policy.', id='policy'),
          Fact(text='New policy.', id='policy'),
      ],
  )

  chunks = await _sent_chunks(plugin, client)

  assert [c.text for c in chunks] == ['New policy.']


def test_facts_needing_more_than_max_facts_chunks_raise_value_error():
  """Facts too large to send in one check are rejected, not dropped."""
  client = _RecordingClient(
      limits=GroundingLimits(max_facts=2, max_fact_chars=4)
  )

  with pytest.raises(ValueError, match='more than max_facts'):
    GroundingPlugin(client=client, facts=[Fact(text='abcdefghij')])


async def test_changing_a_fact_after_construction_does_not_change_what_is_sent():
  """The plugin keeps its own copy of the facts it was given."""
  client = _RecordingClient()
  fact = Fact(text='A fact.')
  plugin = GroundingPlugin(client=client, facts=[fact])

  fact.text = 'Changed.'
  chunks = await _sent_chunks(plugin, client)

  assert [c.text for c in chunks] == ['A fact.']


def test_empty_fact_raises_value_error():
  """Whitespace-only fact text is rejected."""
  with pytest.raises(ValueError, match='must not be empty'):
    Fact(text='  \n')


# Checking responses


async def test_citations_are_attached_to_the_response():
  """Citations from the client become the response's grounding_metadata."""
  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK,
              num_facts_evaluated=1,
              metadata=_METADATA,
          )
      ),
  )

  response, replacement = await _respond(plugin)

  assert replacement is None
  assert response.grounding_metadata == _METADATA


async def test_citations_are_added_to_metadata_the_model_attached():
  """Search metadata from the model is kept, with the citations added."""
  web = types.GroundingMetadata(
      web_search_queries=['boiling point'],
      grounding_chunks=[types.GroundingChunk()],
      grounding_supports=[types.GroundingSupport(grounding_chunk_indices=[0])],
  )
  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK,
              num_facts_evaluated=1,
              metadata=types.GroundingMetadata(
                  grounding_chunks=[types.GroundingChunk()],
                  grounding_supports=[
                      types.GroundingSupport(grounding_chunk_indices=[0])
                  ],
              ),
          )
      ),
  )

  response, _ = await _respond(plugin, metadata=web)

  combined = response.grounding_metadata
  assert combined.web_search_queries == ['boiling point']
  assert len(combined.grounding_chunks) == 2
  assert [s.grounding_chunk_indices for s in combined.grounding_supports] == [
      [0],
      [1],
  ]


async def test_answer_without_citations_gets_no_metadata():
  """A check that cites nothing leaves the response's metadata unset."""
  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK,
              num_facts_evaluated=1,
              metadata=types.GroundingMetadata(),
          )
      ),
  )

  response, _ = await _respond(plugin)

  assert response.grounding_metadata is None


async def test_answer_text_is_sent_unchanged():
  """Surrounding whitespace is kept so citation offsets line up."""
  client = _RecordingClient()
  plugin = GroundingPlugin(facts=[Fact(text='A fact.')], client=client)

  await _respond(plugin, text='\n  An answer.  ')

  assert client.sent[0][0] == '\n  An answer.  '


@pytest.mark.parametrize(
    ('text', 'partial'),
    [('Water boils', True), ('  ', False)],
    ids=['partial', 'blank'],
)
async def test_partial_or_blank_response_is_not_checked(
    text: str, partial: bool
):
  """Streamed partials and responses without text skip the check."""
  client = _RecordingClient()
  plugin = GroundingPlugin(facts=[Fact(text='A fact.')], client=client)

  response, _ = await _respond(plugin, text=text, partial=partial)

  assert client.sent == []
  assert response.grounding_metadata is None


async def test_thoughts_are_not_sent_as_part_of_the_answer():
  """The model's reasoning parts are left out of the checked answer."""
  client = _RecordingClient()
  plugin = GroundingPlugin(facts=[Fact(text='A fact.')], client=client)

  await _respond(
      plugin,
      parts=[
          types.Part(text='Thinking hard.', thought=True),
          types.Part(text='Answer.'),
      ],
  )

  assert client.sent[0][0] == 'Answer.'


async def test_response_that_calls_a_tool_is_not_checked():
  """Text alongside a function call is not the final answer."""
  client = _RecordingClient()
  plugin = GroundingPlugin(facts=[Fact(text='A fact.')], client=client)

  await _respond(
      plugin,
      parts=[
          types.Part(text='Let me look that up.'),
          types.Part(function_call=types.FunctionCall(name='lookup')),
      ],
  )

  assert client.sent == []


def _support(start: int, end: int, text: str) -> types.GroundingMetadata:
  return types.GroundingMetadata(
      grounding_supports=[
          types.GroundingSupport(
              segment=types.Segment(
                  start_index=start, end_index=end, text=text
              ),
              grounding_chunk_indices=[0],
          )
      ]
  )


@pytest.mark.parametrize(
    ('parts', 'support', 'expected'),
    [
        (
            [types.Part(text='Hmm.', thought=True), types.Part(text='Yes.')],
            _support(0, 4, 'Yes.'),
            [(1, 0, 4, 'Yes.')],
        ),
        (
            [types.Part(text='Café. '), types.Part(text='Boils.')],
            _support(7, 13, 'Boils.'),
            [(1, 0, 6, 'Boils.')],
        ),
        (
            [types.Part(text='Water '), types.Part(text='boils.')],
            _support(0, 12, 'Water boils.'),
            [(0, 0, 6, 'Water '), (1, 0, 6, 'boils.')],
        ),
    ],
    ids=['after_thought', 'after_non_ascii', 'spans_parts'],
)
async def test_citation_segments_point_into_the_part_they_cite(
    parts: list[types.Part],
    support: types.GroundingMetadata,
    expected: list[tuple[int, int, int, str]],
):
  """Byte offsets from the joined answer are rebased onto each part."""
  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK, num_facts_evaluated=1, metadata=support
          )
      ),
  )

  response, _ = await _respond(plugin, parts=parts)

  segments = [s.segment for s in response.grounding_metadata.grounding_supports]
  assert [
      (s.part_index, s.start_index, s.end_index, s.text) for s in segments
  ] == expected


async def test_client_error_leaves_the_response_unchanged():
  """A raising client fails open and reports ``API_ERROR`` to the hook."""
  client = _RecordingClient()
  client.check_grounding_async = mock.AsyncMock(
      side_effect=RuntimeError('boom')
  )
  results = []
  plugin = GroundingPlugin(
      client=client,
      facts=[Fact(text='A fact.')],
      on_result=lambda **kw: results.append(kw['grounding_result']),
  )

  response, replacement = await _respond(plugin)

  assert replacement is None
  assert response.grounding_metadata is None
  assert [(r.status, r.error) for r in results] == [
      (GroundingStatus.API_ERROR, 'boom')
  ]


# on_result hook


async def test_on_result_sees_skipped_check_when_no_facts_are_registered():
  """Without facts the client is not called and the hook sees the skip."""
  client = _RecordingClient()
  results = []
  plugin = GroundingPlugin(
      client=client,
      on_result=lambda *, grounding_result, **_: results.append(
          grounding_result.status
      ),
  )

  await _respond(plugin)

  assert client.sent == []
  assert results == [GroundingStatus.SKIPPED_NO_FACTS]


async def test_on_result_sees_the_response_with_citations_attached():
  """The hook receives the response after metadata is attached."""
  seen = []
  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK,
              num_facts_evaluated=1,
              metadata=_METADATA,
          )
      ),
      on_result=lambda *, llm_response, **_: seen.append(
          llm_response.grounding_metadata
      ),
  )

  await _respond(plugin)

  assert seen == [_METADATA]


@pytest.mark.parametrize('is_async', [False, True], ids=['sync', 'async'])
async def test_on_result_can_replace_the_response(is_async: bool):
  """A response returned by the hook, sync or async, replaces the original."""
  fallback = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Unsure.')])
  )

  async def async_hook(**_):
    return fallback

  plugin = GroundingPlugin(
      facts=[Fact(text='A fact.')],
      client=_RecordingClient(),
      on_result=async_hook if is_async else (lambda **_: fallback),
  )

  _, replacement = await _respond(plugin)

  assert replacement is fallback


# Lifecycle


def test_default_client_needs_a_project(monkeypatch):
  """Without a client, the plugin builds a CheckGroundingClient."""
  monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)

  with pytest.raises(ValueError, match='project_id is required'):
    GroundingPlugin()


async def test_close_leaves_a_client_passed_in_open():
  """A client passed to the plugin is left for its owner to close."""
  client = _RecordingClient()
  plugin = GroundingPlugin(client=client)

  await plugin.close()

  assert not client.closed


async def test_close_closes_the_default_client(monkeypatch):
  """The client the plugin built itself is closed with the plugin."""
  monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'test-project')
  close = mock.AsyncMock()
  monkeypatch.setattr(CheckGroundingClient, 'close', close)
  plugin = GroundingPlugin()

  await plugin.close()

  close.assert_awaited_once()


async def test_final_answer_carries_citations_in_a_runner():
  """End to end, the agent's final event carries the grounding metadata."""
  plugin = GroundingPlugin(
      facts=[Fact(text='Water boils at 100 C.')],
      client=_RecordingClient(
          GroundingResult(
              status=GroundingStatus.OK,
              num_facts_evaluated=1,
              metadata=_METADATA,
          )
      ),
  )
  agent = LlmAgent(
      name='assistant',
      model=testing_utils.MockModel.create(responses=['It boils at 100 C.']),
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name='grounded_app', root_agent=agent, plugins=[plugin])
  )

  events = await runner.run_async('When does water boil?')

  assert events[-1].grounding_metadata == _METADATA
