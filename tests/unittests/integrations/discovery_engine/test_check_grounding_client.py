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

"""Tests for CheckGroundingClient."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

from google.adk.integrations.discovery_engine import CheckGroundingClient
from google.adk.integrations.discovery_engine import FactChunk
from google.adk.integrations.discovery_engine import GroundingLimits
from google.adk.integrations.discovery_engine import GroundingStatus
from google.api_core import exceptions as api_exceptions
import google.cloud
from google.cloud import discoveryengine_v1
import pytest

_CLIENT_MODULE = (
    'google.adk.integrations.discovery_engine._check_grounding_client'
)
_FACT = FactChunk(text='Water boils at 100 C.')


def _client_with(
    response: discoveryengine_v1.CheckGroundingResponse | None = None,
    error: Exception | None = None,
    **kwargs,
) -> tuple[CheckGroundingClient, mock.AsyncMock]:
  """Returns a client whose API calls return ``response`` or raise ``error``."""
  check_grounding = mock.AsyncMock(
      return_value=response or discoveryengine_v1.CheckGroundingResponse(),
      side_effect=error,
  )
  api = mock.MagicMock(check_grounding=check_grounding)
  kwargs.setdefault('project_id', 'test-project')
  return CheckGroundingClient(client=api, **kwargs), check_grounding


def _sent_request(
    check_grounding: mock.AsyncMock,
) -> discoveryengine_v1.CheckGroundingRequest:
  return check_grounding.call_args.kwargs['request']


def _cited(*sources: int) -> list[discoveryengine_v1.FactChunk]:
  """Returns cited chunks pointing at the facts at the given request indices."""
  return [discoveryengine_v1.FactChunk(source=str(i)) for i in sources]


def _claim(**kwargs) -> discoveryengine_v1.CheckGroundingResponse.Claim:
  kwargs.setdefault('grounding_check_required', True)
  return discoveryengine_v1.CheckGroundingResponse.Claim(**kwargs)


# Request handling


async def test_no_facts_skips_the_api_call():
  """Checking against no facts returns SKIPPED_NO_FACTS without a call."""
  client, check_grounding = _client_with()

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[]
  )

  assert result.status is GroundingStatus.SKIPPED_NO_FACTS
  check_grounding.assert_not_called()


async def test_blank_answer_skips_the_api_call():
  """A whitespace-only answer returns SKIPPED_EMPTY without a call."""
  client, check_grounding = _client_with()

  result = await client.check_grounding_async(
      answer_candidate='  \n ', facts=[_FACT]
  )

  assert result.status is GroundingStatus.SKIPPED_EMPTY
  assert result.num_facts_evaluated == 0
  check_grounding.assert_not_called()


async def test_request_targets_configured_grounding_config():
  """The request names the configured project, location and config."""
  client, check_grounding = _client_with(
      location='us-central1', grounding_config='custom_config'
  )

  await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  assert _sent_request(check_grounding).grounding_config == (
      'projects/test-project/locations/us-central1/groundingConfigs/'
      'custom_config'
  )


async def test_request_carries_grounding_spec_options():
  """Citation threshold and claim-level scoring reach the request."""
  client, check_grounding = _client_with(
      citation_threshold=0.5, enable_claim_level_score=False
  )

  await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  spec = _sent_request(check_grounding).grounding_spec
  assert spec.citation_threshold == 0.5
  assert spec.enable_claim_level_score is False


async def test_api_call_has_a_per_attempt_timeout():
  """Each call is bounded so a hung request cannot block the turn."""
  client, check_grounding = _client_with()

  await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  timeout = check_grounding.call_args.kwargs.get('timeout')
  assert timeout is not None and timeout > 0


async def test_fact_title_link_and_attributes_are_sent_as_attributes():
  """Fact metadata is sent so the API can return it with citations."""
  client, check_grounding = _client_with()
  fact = FactChunk(
      text='Evidence.',
      title='Doc',
      link='https://docs.example.com',
      attributes={'department': 'security'},
  )

  await client.check_grounding_async(
      answer_candidate='An answer.', facts=[fact]
  )

  sent = _sent_request(check_grounding).facts[0]
  assert sent.fact_text == 'Evidence.'
  assert dict(sent.attributes) == {
      'department': 'security',
      'title': 'Doc',
      'source': 'https://docs.example.com',
  }


async def test_caller_attributes_are_not_overwritten_by_title_or_link():
  """A caller's own ``title`` or ``source`` attribute is kept as given."""
  client, check_grounding = _client_with()
  fact = FactChunk(
      text='Evidence.',
      title='Doc',
      link='https://docs.example.com',
      attributes={'title': 'Ticket 42', 'source': 'crm'},
  )

  await client.check_grounding_async(
      answer_candidate='An answer.', facts=[fact]
  )

  sent = _sent_request(check_grounding).facts[0]
  assert dict(sent.attributes) == {'title': 'Ticket 42', 'source': 'crm'}


async def test_only_the_first_max_facts_are_sent():
  """Facts beyond ``limits.max_facts`` are dropped from the request."""
  client, check_grounding = _client_with(limits=GroundingLimits(max_facts=2))
  facts = [FactChunk(text=f'Fact {i}.') for i in range(5)]

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=facts
  )

  sent = [f.fact_text for f in _sent_request(check_grounding).facts]
  assert sent == ['Fact 0.', 'Fact 1.']
  assert result.num_facts_evaluated == 2


# Error handling


@pytest.mark.parametrize(
    ('error', 'expected_status'),
    [
        (
            api_exceptions.InvalidArgument(
                'The answer candidate exceeds the maximum token limit of 8192.'
            ),
            GroundingStatus.REJECTED_TOO_LONG,
        ),
        (
            api_exceptions.InvalidArgument('malformed grounding_config'),
            GroundingStatus.API_ERROR,
        ),
        (
            api_exceptions.ServiceUnavailable('backend down'),
            GroundingStatus.API_ERROR,
        ),
    ],
    ids=['token_limit', 'other_invalid_argument', 'other_api_error'],
)
async def test_api_failure_is_reported_not_raised(
    error: Exception, expected_status: GroundingStatus
):
  """API failures map to a status and message instead of raising."""
  client, _ = _client_with(error=error)

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  assert result.status is expected_status
  assert result.error == str(error)
  assert result.metadata is None


async def test_non_api_error_is_raised():
  """An error that did not come from the API reaches the caller."""
  client, _ = _client_with(error=RuntimeError('boom'))

  with pytest.raises(RuntimeError, match='boom'):
    await client.check_grounding_async(
        answer_candidate='An answer.', facts=[_FACT]
    )


# Response mapping


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        (discoveryengine_v1.CheckGroundingResponse(support_score=0.5), 0.5),
        (discoveryengine_v1.CheckGroundingResponse(), None),
    ],
    ids=['present', 'missing'],
)
async def test_support_score_is_reported_only_when_present(
    response: discoveryengine_v1.CheckGroundingResponse, expected: float | None
):
  """The support score is passed through, and None when the API omits it."""
  client, _ = _client_with(response)

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  assert result.status is GroundingStatus.OK
  assert result.support_score == expected


async def test_only_claims_requiring_a_check_are_counted():
  """Claims such as greetings do not count toward ``num_claims``."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          claims=[
              _claim(citation_indices=[0]),
              _claim(citation_indices=[]),
              _claim(citation_indices=[0], grounding_check_required=False),
          ],
      )
  )

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  assert result.num_claims == 2


async def test_claim_maps_to_support_with_segment_and_citations():
  """A cited claim becomes a support with its segment and chunk indices."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=_cited(0, 1),
          claims=[
              _claim(
                  start_pos=0,
                  end_pos=14,
                  claim_text='Statement one.',
                  citation_indices=[1],
                  score=0.95,
              )
          ],
      )
  )
  facts = [FactChunk(text='Fact A.'), FactChunk(text='Fact B.')]

  result = await client.check_grounding_async(
      answer_candidate='Statement one.', facts=facts
  )

  support = result.metadata.grounding_supports[0]
  assert (support.segment.start_index, support.segment.end_index) == (0, 14)
  assert support.segment.text == 'Statement one.'
  assert support.grounding_chunk_indices == [1]
  assert support.confidence_scores == [0.95]


async def test_claim_not_requiring_check_is_not_cited():
  """Claims such as greetings are left out of the supports."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=_cited(0),
          claims=[
              _claim(
                  claim_text='Hello!',
                  citation_indices=[0],
                  grounding_check_required=False,
              ),
              _claim(claim_text='Water boils at 100 C.', citation_indices=[0]),
          ],
      )
  )

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  texts = [s.segment.text for s in result.metadata.grounding_supports]
  assert texts == ['Water boils at 100 C.']


@pytest.mark.parametrize(
    ('support_score', 'claim_score', 'expected'),
    [
        (0.5, 0.25, [0.25, 0.25]),
        (0.5, 0.0, [0.0, 0.0]),
        (0.5, None, [0.5, 0.5]),
        (None, None, None),
    ],
    ids=['claim_score', 'explicit_zero', 'support_fallback', 'unset'],
)
async def test_confidence_comes_from_claim_then_support_score(
    support_score: float | None,
    claim_score: float | None,
    expected: list[float] | None,
):
  """Each citation gets the claim score, else the support score, else none."""
  response = discoveryengine_v1.CheckGroundingResponse(
      cited_chunks=_cited(0, 1),
      claims=[
          _claim(citation_indices=[0, 1])
          if claim_score is None
          else _claim(citation_indices=[0, 1], score=claim_score)
      ],
  )
  if support_score is not None:
    response.support_score = support_score
  client, _ = _client_with(response)

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT, _FACT]
  )

  assert result.metadata.grounding_supports[0].confidence_scores == expected


async def test_cited_chunk_fills_missing_fields_from_the_fact_it_cites():
  """Text, title, link and attributes come from the fact at ``source``."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=_cited(1), claims=[_claim(citation_indices=[0])]
      )
  )
  facts = [
      FactChunk(text='Other fact.'),
      FactChunk(
          text='The sky is blue.',
          title='Optics',
          link='https://example.com/optics',
          attributes={'topic': 'physics'},
      ),
  ]

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=facts
  )

  context = result.metadata.grounding_chunks[0].retrieved_context
  assert (context.text, context.title, context.uri) == (
      'The sky is blue.',
      'Optics',
      'https://example.com/optics',
  )
  assert [(m.key, m.string_value) for m in context.custom_metadata] == [
      ('topic', 'physics')
  ]


async def test_citation_index_outside_cited_chunks_is_dropped():
  """A claim citing past the end of ``cited_chunks`` gets no support."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=_cited(0), claims=[_claim(citation_indices=[1])]
      )
  )

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT]
  )

  assert result.metadata.grounding_supports is None


async def test_cited_chunk_with_unknown_source_keeps_its_own_text():
  """A cited chunk whose source is not a fact index is not enriched."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=[
              discoveryengine_v1.FactChunk(
                  chunk_text='Quoted text.', source='doc-7'
              )
          ],
          claims=[_claim(citation_indices=[0])],
      )
  )
  fact = FactChunk(text='Fact.', title='Doc')

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[fact]
  )

  context = result.metadata.grounding_chunks[0].retrieved_context
  assert (context.text, context.title) == ('Quoted text.', None)


async def test_cited_chunk_with_non_ascii_digit_source_is_not_enriched():
  """A Unicode digit such as '²' is not read as a fact index."""
  client, _ = _client_with(
      discoveryengine_v1.CheckGroundingResponse(
          cited_chunks=[
              discoveryengine_v1.FactChunk(
                  chunk_text='Quoted text.', source='²'
              )
          ],
          claims=[_claim(citation_indices=[0])],
      )
  )

  result = await client.check_grounding_async(
      answer_candidate='An answer.', facts=[_FACT, _FACT, _FACT]
  )

  context = result.metadata.grounding_chunks[0].retrieved_context
  assert context.text == 'Quoted text.'


# Client construction


@pytest.mark.parametrize(
    ('location', 'expected_endpoint'),
    [
        ('global', None),
        ('us-central1', 'us-central1-discoveryengine.googleapis.com'),
    ],
)
def test_auto_created_client_endpoint_matches_location(
    monkeypatch, location: str, expected_endpoint: str | None
):
  """Regional locations use the regional endpoint; global uses the default."""
  # A client certificate on the machine would otherwise select mTLS endpoints.
  monkeypatch.setenv('GOOGLE_API_USE_MTLS_ENDPOINT', 'never')
  created = mock.MagicMock()
  monkeypatch.setattr(
      discoveryengine_v1, 'GroundedGenerationServiceAsyncClient', created
  )
  client = CheckGroundingClient(project_id='test-project', location=location)

  _ = client.client

  options = created.call_args.kwargs['client_options']
  assert (options.api_endpoint if options else None) == expected_endpoint


def test_limits_are_exposed_for_callers():
  """The configured limits are readable, for callers sizing their facts."""
  limits = GroundingLimits(max_facts=5, max_fact_chars=100)

  client = CheckGroundingClient(project_id='test-project', limits=limits)

  assert client.limits == limits


def test_missing_project_raises_value_error(monkeypatch):
  """Construction fails clearly when no project is configured."""
  monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)

  with pytest.raises(ValueError, match='project_id is required'):
    CheckGroundingClient()


@pytest.mark.parametrize('threshold', [-0.1, 1.5])
def test_citation_threshold_outside_unit_range_raises_value_error(
    threshold: float,
):
  """A threshold outside 0 to 1 is rejected at construction."""
  with pytest.raises(ValueError, match='citation_threshold'):
    CheckGroundingClient(
        project_id='test-project', citation_threshold=threshold
    )


def test_missing_discoveryengine_names_the_extra_to_install(monkeypatch):
  """Importing without the dependency says which extra to install."""
  # `from google.cloud import X` short-circuits on the parent attribute.
  monkeypatch.delattr(google.cloud, 'discoveryengine_v1', raising=False)
  monkeypatch.setitem(sys.modules, 'google.cloud.discoveryengine_v1', None)
  monkeypatch.delitem(sys.modules, _CLIENT_MODULE)

  with pytest.raises(ImportError, match=r'google-adk\[gcp\]'):
    importlib.import_module(_CLIENT_MODULE)


def test_missing_api_core_names_the_extra_to_install(monkeypatch):
  """Importing without google-api-core says which extra to install."""
  monkeypatch.setitem(sys.modules, 'google.api_core', None)
  monkeypatch.delitem(sys.modules, _CLIENT_MODULE)

  with pytest.raises(ImportError, match=r'google-adk\[gcp\]'):
    importlib.import_module(_CLIENT_MODULE)


async def test_close_closes_the_auto_created_client(monkeypatch):
  """Closing releases the transport of the client built for this loop."""
  created = mock.MagicMock()
  created.return_value.transport.close = mock.AsyncMock()
  monkeypatch.setattr(
      discoveryengine_v1, 'GroundedGenerationServiceAsyncClient', created
  )
  client = CheckGroundingClient(project_id='test-project')
  _ = client.client

  await client.close()

  created.return_value.transport.close.assert_awaited_once()


async def test_close_leaves_a_supplied_client_open():
  """A client passed to the constructor is left for its owner to close."""
  api = mock.MagicMock()
  api.transport.close = mock.AsyncMock()
  client = CheckGroundingClient(project_id='test-project', client=api)

  await client.close()

  api.transport.close.assert_not_awaited()
