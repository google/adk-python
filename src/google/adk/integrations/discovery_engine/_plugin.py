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

"""Plugin that verifies model responses against registered facts."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Sequence
import logging
import re
from typing import Protocol

from google.genai import types
from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...features import experimental
from ...features import FeatureName
from ...models.llm_response import LlmResponse
from ...plugins.base_plugin import BasePlugin
from ._types import Fact
from ._types import FactChunk
from ._types import GroundingLimits
from ._types import GroundingResult
from ._types import GroundingStatus

logger = logging.getLogger('google_adk.' + __name__)

# Tried in order, so a chunk ends at the coarsest boundary that fits.
_BOUNDARIES = (
    re.compile(r'\n\s*\n'),
    re.compile(r'\n'),
    re.compile(r'(?<=[.!?])\s+'),
    re.compile(r'\s+'),
)


class GroundingClient(Protocol):
  """A backend that checks how well facts support an answer."""

  @property
  def limits(self) -> GroundingLimits:
    """Capacity limits of the backend."""
    ...

  async def check_grounding_async(
      self, *, answer_candidate: str, facts: Sequence[FactChunk]
  ) -> GroundingResult:
    """Checks the answer against the facts.

    Args:
      answer_candidate: The model response text to verify.
      facts: Evidence to verify against.

    Returns:
      The outcome of the check.
    """
    ...

  async def close(self) -> None:
    """Releases resources the client holds."""
    ...


class OnResult(Protocol):
  """Hook called with the result of every grounding check.

  Return ``None`` to keep the model response, which already carries any
  grounding metadata, or an ``LlmResponse`` to replace it. Replacing it skips
  later plugins and agent ``after_model_callback`` hooks, as in ADK's own
  ``after_model_callback``.

  When the agent has a tool with ``propagate_grounding_metadata=True``, ADK
  replaces the response's grounding metadata with that tool's after this hook
  runs, so ``grounding_result`` is the only place the check's citations appear.
  """

  def __call__(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
      grounding_result: GroundingResult,
  ) -> Awaitable[LlmResponse | None] | LlmResponse | None:
    ...


@experimental(FeatureName.CHECK_GROUNDING)
class GroundingPlugin(BasePlugin):
  """Verifies final model responses against a fixed set of facts.

  For every final text response, the plugin sends the response and the facts
  to a ``GroundingClient``, then attaches the returned citations to the
  response as ``grounding_metadata``. A failed check leaves the response
  unchanged.
  """

  def __init__(
      self,
      *,
      client: GroundingClient | None = None,
      facts: Sequence[Fact] = (),
      on_result: OnResult | None = None,
      name: str = 'grounding_plugin',
  ) -> None:
    """Initializes the plugin.

    Facts are fixed here and apply to every session and user the plugin
    serves, so pass only knowledge shared by all of them.

    Args:
      client: The grounding backend. Defaults to a ``CheckGroundingClient``
        for the ``GOOGLE_CLOUD_PROJECT`` project.
      facts: Facts every response is checked against. Text longer than the
        client's ``limits.max_fact_chars`` is split into several chunks. A
        later fact with the same ``id`` replaces an earlier one.
      on_result: Optional hook called with each check's result, including
        skipped and failed checks.
      name: Name of this plugin instance.

    Raises:
      ValueError: If the facts need more chunks than ``limits.max_facts``.
    """
    super().__init__(name)
    # Only a client built here is this plugin's to close.
    self._owns_client = client is None
    if client is None:
      # Imported here so custom backends do not need the Discovery Engine SDK.
      from ._check_grounding_client import CheckGroundingClient

      client = CheckGroundingClient()
    self._client = client
    self._on_result = on_result
    limits = client.limits
    by_id = {fact.id: fact for fact in facts}
    self._chunks = tuple(
        chunk
        for fact in by_id.values()
        for chunk in _to_chunks(fact, limits.max_fact_chars)
    )
    if len(self._chunks) > limits.max_facts:
      raise ValueError(
          f'Facts need {len(self._chunks)} chunks, more than max_facts'
          f' ({limits.max_facts}).'
      )

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    """Checks a final text response and attaches its citations."""
    # Streamed partial responses would yield offsets into incomplete text.
    if llm_response.partial or not llm_response.content:
      return None
    parts = llm_response.content.parts or []
    # A response that calls a tool is not the final answer.
    if any(p.function_call for p in parts):
      return None
    # Thoughts are the model's reasoning, not part of the answer.
    answer_parts = [
        (i, p.text) for i, p in enumerate(parts) if p.text and not p.thought
    ]
    # Not stripped: citation offsets index into this exact text.
    answer = ''.join(text for _, text in answer_parts)
    if not answer.strip():
      return None

    if self._chunks:
      result = await self._check(answer, self._chunks)
    else:
      result = GroundingResult(
          status=GroundingStatus.SKIPPED_NO_FACTS, num_facts_evaluated=0
      )
    logger.debug(
        'Grounding check finished with status %s for invocation %s',
        result.status,
        callback_context.invocation_id,
    )
    if result.metadata is not None:
      citations = _map_to_parts(result.metadata, answer_parts)
      result = result.model_copy()
      result.metadata = citations
      # Empty metadata is truthy, so attaching it would mark an uncited answer
      # as grounded.
      if citations.grounding_chunks or citations.grounding_supports:
        llm_response.grounding_metadata = _combine(
            llm_response.grounding_metadata, citations
        )

    if self._on_result is None:
      return None
    replacement = self._on_result(
        callback_context=callback_context,
        llm_response=llm_response,
        grounding_result=result,
    )
    if isinstance(replacement, Awaitable):
      replacement = await replacement
    return replacement

  async def _check(
      self, answer: str, facts: Sequence[FactChunk]
  ) -> GroundingResult:
    """Runs the check, reporting any client error as ``API_ERROR``."""
    try:
      return await self._client.check_grounding_async(
          answer_candidate=answer, facts=facts
      )
    # A verification failure must not fail the turn; on_result still sees it.
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning(
          'Grounding check failed; response left unchanged.', exc_info=True
      )
      return GroundingResult(
          status=GroundingStatus.API_ERROR,
          num_facts_evaluated=len(facts),
          error=str(e),
      )

  @override
  async def close(self) -> None:
    """Closes the default client; a client passed in is left to its owner."""
    if self._owns_client:
      await self._client.close()


def _map_to_parts(
    metadata: types.GroundingMetadata, answer_parts: list[tuple[int, str]]
) -> types.GroundingMetadata:
  """Moves support segments from the joined answer onto its parts.

  The client checks the answer parts joined into one string, but a ``Segment``
  holds byte offsets within the part that ``part_index`` names. A support that
  spans parts becomes one support per part.

  Args:
    metadata: Metadata whose segments index the joined answer.
    answer_parts: ``(part_index, text)`` of each part in the joined answer.

  Returns:
    The metadata with each segment rebased onto its part.
  """
  joined = ''.join(text for _, text in answer_parts).encode()
  ranges = []
  start = 0
  for index, text in answer_parts:
    end = start + len(text.encode())
    ranges.append((index, start, end))
    start = end
  supports = []
  for support in metadata.grounding_supports or []:
    segment = support.segment
    if (
        segment is None
        or segment.start_index is None
        or segment.end_index is None
    ):
      supports.append(support)
      continue
    for index, part_start, part_end in ranges:
      low = max(segment.start_index, part_start)
      high = min(segment.end_index, part_end)
      if low >= high:
        continue
      whole = (low, high) == (segment.start_index, segment.end_index)
      mapped = support.model_copy()
      mapped.segment = types.Segment(
          part_index=index,
          start_index=low - part_start,
          end_index=high - part_start,
          text=segment.text
          if whole
          else joined[low:high].decode(errors='replace'),
      )
      supports.append(mapped)
  rebased = metadata.model_copy()
  rebased.grounding_supports = supports or None
  return rebased


def _combine(
    existing: types.GroundingMetadata | None, added: types.GroundingMetadata
) -> types.GroundingMetadata:
  """Adds the check's citations to metadata the model already attached.

  The added chunks go after the existing ones, so the added supports' chunk
  indices shift by the number of existing chunks. Other fields, such as search
  queries, are kept from the existing metadata.

  Args:
    existing: Metadata on the response before the check, if any.
    added: Citations from the check.

  Returns:
    The combined metadata.
  """
  if existing is None:
    return added
  offset = len(existing.grounding_chunks or [])
  shifted = []
  for support in added.grounding_supports or []:
    moved = support.model_copy()
    moved.grounding_chunk_indices = [
        i + offset for i in support.grounding_chunk_indices or []
    ]
    shifted.append(moved)
  combined = existing.model_copy()
  combined.grounding_chunks = [
      *(existing.grounding_chunks or []),
      *(added.grounding_chunks or []),
  ] or None
  combined.grounding_supports = [
      *(existing.grounding_supports or []),
      *shifted,
  ] or None
  return combined


def _to_chunks(fact: Fact, max_chars: int) -> list[FactChunk]:
  return [
      FactChunk(
          text=text,
          title=fact.title,
          link=fact.link,
          attributes=dict(fact.attributes),
      )
      for text in _split_fact_text(fact.text, max_chars)
  ]


def _split_fact_text(text: str, max_chars: int) -> list[str]:
  """Splits text into chunks of at most ``max_chars`` characters.

  Chunks end at paragraph, line, sentence or word boundaries where one falls
  inside the limit, and are cut at exactly ``max_chars`` otherwise.

  Args:
    text: The text to split.
    max_chars: Maximum length of each chunk.

  Returns:
    The non-empty chunks, stripped of surrounding whitespace.
  """
  chunks = []
  remaining = text.strip()
  while len(remaining) > max_chars:
    window = remaining[:max_chars]
    end = max_chars
    for boundary in _BOUNDARIES:
      matches = list(boundary.finditer(window))
      if matches:
        end = matches[-1].end()
        break
    chunk = remaining[:end].strip()
    if chunk:
      chunks.append(chunk)
    remaining = remaining[end:].lstrip()
  if remaining:
    chunks.append(remaining)
  return chunks
