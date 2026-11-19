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

"""Client for the Discovery Engine CheckGrounding API."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os

from google.genai import types

from ... import version
from ...features import experimental
from ...features import FeatureName
from ...utils._event_loop_cache import per_loop_value
from ...utils._mtls_utils import get_api_endpoint
from .._google_sdk import create_gapic_client_info as _create_gapic_client_info
from ._types import FactChunk
from ._types import GroundingLimits
from ._types import GroundingResult
from ._types import GroundingStatus

try:
  from google.api_core import exceptions as api_exceptions
  from google.api_core.client_options import ClientOptions
  from google.api_core.retry import if_exception_type
  from google.api_core.retry_async import AsyncRetry
  from google.cloud import discoveryengine_v1
except ImportError as e:
  raise ImportError(
      'CheckGrounding support requires google-cloud-discoveryengine. '
      "Install it with: pip install 'google-adk[gcp]'."
  ) from e

logger = logging.getLogger('google_adk.' + __name__)

USER_AGENT = f'adk-check-grounding google-adk/{version.__version__}'

_GLOBAL_LOCATION = 'global'

# 429, 503 and 504 are transient. Retrying them with bounded backoff keeps a
# brief quota or availability blip from dropping a turn's citations.
_DEFAULT_RETRY = AsyncRetry(
    predicate=if_exception_type(
        api_exceptions.ResourceExhausted,
        api_exceptions.ServiceUnavailable,
        api_exceptions.DeadlineExceeded,
    ),
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    timeout=60.0,
)

# The GAPIC method has no default deadline, so a hung call would block the
# turn. Bounds each attempt; _DEFAULT_RETRY bounds the total.
_DEFAULT_TIMEOUT = 20.0


@experimental(FeatureName.CHECK_GROUNDING)
class CheckGroundingClient:
  """Verifies answers against facts with the Discovery Engine CheckGrounding API.

  The API rejects answers longer than its token limit; such answers produce a
  ``REJECTED_TOO_LONG`` result rather than an exception.
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str = _GLOBAL_LOCATION,
      grounding_config: str = 'default_grounding_config',
      citation_threshold: float = 0.6,
      enable_claim_level_score: bool = True,
      limits: GroundingLimits | None = None,
      client: (
          discoveryengine_v1.GroundedGenerationServiceAsyncClient | None
      ) = None,
  ) -> None:
    """Initializes the client.

    Args:
      project_id: Google Cloud project ID. Defaults to the
        ``GOOGLE_CLOUD_PROJECT`` environment variable.
      location: Discovery Engine location of the grounding config.
      grounding_config: Name of the grounding config resource.
      citation_threshold: Minimum confidence, from 0 to 1, for a fact to be
        cited.
      enable_claim_level_score: Whether to request a score for each claim.
      limits: Capacity limits of the backend. Defaults to the API limits.
      client: Optional pre-constructed async client, for example one with
        custom credentials. If ``None``, one is constructed for each event loop
        on first use with Application Default Credentials.

    Raises:
      ValueError: If no project ID can be resolved, or if
        ``citation_threshold`` is outside 0 to 1.
    """
    resolved_project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
    if not resolved_project:
      raise ValueError(
          'project_id is required; pass it or set GOOGLE_CLOUD_PROJECT.'
      )
    if not 0.0 <= citation_threshold <= 1.0:
      raise ValueError(
          'citation_threshold must be between 0 and 1, got'
          f' {citation_threshold}.'
      )
    self._project_id = resolved_project
    self._location = location
    self._grounding_config = grounding_config
    self._citation_threshold = citation_threshold
    self._enable_claim_level_score = enable_claim_level_score
    self._limits = limits or GroundingLimits()
    self._supplied_client = client

  @property
  def client(self) -> discoveryengine_v1.GroundedGenerationServiceAsyncClient:
    """Returns the async client for the running event loop.

    A ``grpc.aio`` channel binds to the event loop running when it is created,
    so every loop gets its own client, constructed on first use.
    """
    if self._supplied_client:
      return self._supplied_client
    clients = self._clients_on_running_loop()
    if not clients:
      clients.append(self._create_client())
    return clients[0]

  async def close(self) -> None:
    """Closes the client built for the running event loop.

    A client passed to the constructor is left open for its owner to close.
    """
    clients = self._clients_on_running_loop()
    if clients:
      transport = clients.pop().transport
      await transport.close()  # type: ignore[no-untyped-call]  # GAPIC transports are untyped

  def _clients_on_running_loop(
      self,
  ) -> list[discoveryengine_v1.GroundedGenerationServiceAsyncClient]:
    return per_loop_value(self, '_clients_per_loop', list)

  def _create_client(
      self,
  ) -> discoveryengine_v1.GroundedGenerationServiceAsyncClient:
    client_options = None
    if self._location != _GLOBAL_LOCATION:
      client_options = ClientOptions(
          api_endpoint=get_api_endpoint(
              location=self._location,
              default_template='{location}-discoveryengine.googleapis.com',
              mtls_template='{location}-discoveryengine.mtls.googleapis.com',
          )
      )
    return discoveryengine_v1.GroundedGenerationServiceAsyncClient(
        client_info=_create_gapic_client_info(user_agent=USER_AGENT),
        client_options=client_options,
    )

  async def check_grounding_async(
      self,
      *,
      answer_candidate: str,
      facts: Sequence[FactChunk],
  ) -> GroundingResult:
    """Checks how well the facts support an answer.

    Args:
      answer_candidate: The model response text to verify.
      facts: Evidence to verify against. Only the first ``limits.max_facts``
        chunks are sent.

    Returns:
      The outcome of the check. Errors returned by the API are reported
      through the result's status rather than raised.

    Raises:
      google.auth.exceptions.GoogleAuthError: If the client is built here and
        no credentials can be found.
    """
    if not facts:
      return GroundingResult(
          status=GroundingStatus.SKIPPED_NO_FACTS, num_facts_evaluated=0
      )
    facts = facts[: self._limits.max_facts]
    if not answer_candidate.strip():
      return GroundingResult(
          status=GroundingStatus.SKIPPED_EMPTY, num_facts_evaluated=0
      )

    grounding_config = (
        f'projects/{self._project_id}/locations/{self._location}'
        f'/groundingConfigs/{self._grounding_config}'
    )
    request = discoveryengine_v1.CheckGroundingRequest(
        grounding_config=grounding_config,
        answer_candidate=answer_candidate,
        facts=[_to_grounding_fact(chunk) for chunk in facts],
        grounding_spec=discoveryengine_v1.CheckGroundingSpec(
            citation_threshold=self._citation_threshold,
            enable_claim_level_score=self._enable_claim_level_score,
        ),
    )
    # Built outside the try so that missing credentials raise instead of
    # being reported as an API error on every call.
    client = self.client
    try:
      response = await client.check_grounding(
          request=request, retry=_DEFAULT_RETRY, timeout=_DEFAULT_TIMEOUT
      )
    except api_exceptions.InvalidArgument as e:
      status = (
          GroundingStatus.REJECTED_TOO_LONG
          if _is_answer_too_long(e)
          else GroundingStatus.API_ERROR
      )
      logger.warning(
          'CheckGrounding rejected the request for %s: %s', grounding_config, e
      )
      return GroundingResult(
          status=status, num_facts_evaluated=len(facts), error=str(e)
      )
    except api_exceptions.GoogleAPIError as e:
      logger.warning(
          'CheckGrounding request for %s failed: %s', grounding_config, e
      )
      return GroundingResult(
          status=GroundingStatus.API_ERROR,
          num_facts_evaluated=len(facts),
          error=str(e),
      )

    return GroundingResult(
        status=GroundingStatus.OK,
        num_facts_evaluated=len(facts),
        metadata=_to_grounding_metadata(response, facts),
        # Unset optional proto floats read as 0.0, so check presence instead.
        support_score=(
            response.support_score if 'support_score' in response else None
        ),
        num_claims=sum(
            1 for c in response.claims if c.grounding_check_required
        ),
    )


def _is_answer_too_long(error: api_exceptions.InvalidArgument) -> bool:
  """Reports whether the API rejected the answer for its token count.

  The API reports this with a generic InvalidArgument whose message names the
  token limit, so the message is the only way to tell it apart.
  """
  return 'token limit' in str(error).lower()


def _to_grounding_fact(chunk: FactChunk) -> discoveryengine_v1.GroundingFact:
  attributes = dict(chunk.attributes)
  if chunk.title:
    attributes.setdefault('title', chunk.title)
  if chunk.link:
    attributes.setdefault('source', chunk.link)
  return discoveryengine_v1.GroundingFact(
      fact_text=chunk.text, attributes=attributes
  )


def _to_grounding_chunks(
    response: discoveryengine_v1.CheckGroundingResponse,
    facts: Sequence[FactChunk],
) -> list[types.GroundingChunk]:
  """Builds citation chunks, enriched from the facts the response cites."""
  chunks = []
  for cited in response.cited_chunks:
    # ``source`` holds the index of the cited fact in the request. ``isdigit``
    # alone accepts Unicode digits such as '²' that ``int`` rejects.
    fact = None
    source = cited.source
    if source.isascii() and source.isdigit() and int(source) < len(facts):
      fact = facts[int(source)]
    attributes = dict(cited.source_metadata) or (
        fact.attributes if fact else {}
    )
    chunks.append(
        types.GroundingChunk(
            retrieved_context=types.GroundingChunkRetrievedContext(
                text=cited.chunk_text or (fact.text if fact else None),
                title=cited.title or (fact.title if fact else None),
                uri=cited.uri or (fact.link if fact else None),
                custom_metadata=[
                    types.GroundingChunkCustomMetadata(key=k, string_value=v)
                    for k, v in attributes.items()
                ]
                or None,
            )
        )
    )
  return chunks


def _to_grounding_metadata(
    response: discoveryengine_v1.CheckGroundingResponse,
    facts: Sequence[FactChunk],
) -> types.GroundingMetadata:
  """Maps a CheckGrounding response onto ADK grounding metadata."""
  # Claims cite by index into ``cited_chunks``.
  chunks = _to_grounding_chunks(response, facts)
  supports = []
  for claim in response.claims:
    if not claim.grounding_check_required:
      continue
    indices = [i for i in claim.citation_indices if 0 <= i < len(chunks)]
    if not indices:
      continue
    # Unset optional proto floats read as 0.0, so check presence instead.
    confidence_scores = None
    if 'score' in claim:
      confidence_scores = [round(claim.score, 4)] * len(indices)
    elif 'support_score' in response:
      confidence_scores = [round(response.support_score, 4)] * len(indices)
    supports.append(
        types.GroundingSupport(
            segment=types.Segment(
                start_index=claim.start_pos if 'start_pos' in claim else None,
                end_index=claim.end_pos if 'end_pos' in claim else None,
                text=claim.claim_text or None,
            ),
            grounding_chunk_indices=indices,
            confidence_scores=confidence_scores,
        )
    )
  return types.GroundingMetadata(
      grounding_chunks=chunks or None,
      grounding_supports=supports or None,
  )
