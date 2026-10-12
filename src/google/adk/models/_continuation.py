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

"""Resumes a generation that the model paused with a continuation token."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import itertools
import logging

from google.genai import types

logger = logging.getLogger('google_adk.' + __name__)

_TEXT_FIELDS = frozenset({'text', 'thought', 'thought_signature'})
_MAX_RESUMES = 256
# A resumed request carries the output so far, which one failure would discard.
_RESUME_RETRY_OPTIONS = types.HttpRetryOptions(attempts=8)
# A resume resends the prompt, so these repeat on every request and describe the
# turn only when taken from the first one.
_PROMPT_USAGE_FIELDS = frozenset({
    'cache_tokens_details',
    'cached_content_token_count',
    'prompt_token_count',
    'prompt_tokens_details',
})
_TOTALLED_USAGE_FIELDS = (
    'prompt_token_count',
    'candidates_token_count',
    'thoughts_token_count',
    'tool_use_prompt_token_count',
)


def continuation_token(
    response: types.GenerateContentResponse,
) -> bytes | None:
  """Returns the token that resumes ``response``, or None if it is complete."""
  if not response.candidates:
    return None
  candidate = response.candidates[0]
  if candidate.finish_reason != types.FinishReason.CONTINUATION:
    return None
  return candidate.continuation_token or None


class Continuation:
  """Carries one generation across the requests that resume it."""

  def __init__(
      self,
      contents: Sequence[types.Content],
      config: types.GenerateContentConfig | None,
      *,
      retry_resends: bool = False,
  ) -> None:
    self._contents = contents
    self._config = config
    self._retry_resends = retry_resends
    self._parts: list[types.Part] = []
    self._usage: types.GenerateContentResponseUsageMetadata | None = None
    self._token: bytes | None = None
    self._resumes = 0

  @property
  def resumed(self) -> bool:
    """Whether the generation took more than one request."""
    return self._token is not None

  @property
  def content(self) -> types.Content | None:
    """Everything generated across the requests recorded so far."""
    if not self._parts:
      return None
    return types.Content(role='model', parts=list(self._parts))

  @property
  def usage(self) -> types.GenerateContentResponseUsageMetadata | None:
    """The token usage summed over the requests recorded so far."""
    return self._usage

  def will_resume(self, token: bytes) -> bool:
    """Whether a request that ended with ``token`` is resumed."""
    return token != self._token and self._resumes < _MAX_RESUMES

  def advance(
      self,
      *,
      token: bytes | None,
      parts: Sequence[types.Part] | None,
      usage: types.GenerateContentResponseUsageMetadata | None,
  ) -> tuple[list[types.Content], types.GenerateContentConfig] | None:
    """Records one request's output and returns the request that resumes it.

    Args:
      token: The continuation token the request ended with, if any.
      parts: The parts the request generated.
      usage: The token usage the request reported.

    Returns:
      The contents and config of the next request, or None once the
      generation is complete.
    """
    self._usage = _add_usage(self._usage, usage)
    if token is None and not self.resumed:
      return None
    for index, part in enumerate(parts or []):
      _append_part(self._parts, part, at_pause=index == 0)
    if token is None:
      return None
    if ends_mid_call(self._parts):
      logger.warning(
          'The model paused in the middle of a function call, which cannot be'
          ' resumed; returning the output generated so far.'
      )
      return None
    if not self.will_resume(token):
      if token == self._token:
        # A token that did not change means the model made no progress.
        logger.warning(
            'The model returned the same continuation token twice; returning'
            ' the output generated so far.'
        )
      else:
        logger.warning(
            'The model paused the generation %d times; returning the output'
            ' generated so far.',
            _MAX_RESUMES,
        )
      return None
    self._token = token
    self._resumes += 1
    logger.info('The model paused the generation; resuming it.')
    return self._next_request(token)

  def _next_request(
      self, token: bytes
  ) -> tuple[list[types.Content], types.GenerateContentConfig]:
    """Returns the contents and config of the request that resumes ``token``."""
    contents = list(self._contents)
    if self._parts:
      contents.append(types.Content(role='model', parts=list(self._parts)))
    config = (
        self._config.model_copy(update={'continuation_token': token})
        if self._config
        else types.GenerateContentConfig(continuation_token=token)
    )
    if self._retry_resends and (
        config.http_options is None or config.http_options.retry_options is None
    ):
      config.http_options = (
          config.http_options or types.HttpOptions()
      ).model_copy(update={'retry_options': _RESUME_RETRY_OPTIONS})
    return contents, config


def ends_mid_call(parts: Sequence[types.Part]) -> bool:
  """Whether ``parts`` ends with a function call the model has not finished.

  Streaming a call's arguments splits it over parts that set ``will_continue``
  until the last one, and a request whose final model part still sets it is
  rejected, so a generation paused there cannot be resumed.
  """
  if not parts:
    return False
  function_call = parts[-1].function_call
  return bool(function_call and function_call.will_continue)


def join_parts(parts: Sequence[types.Part]) -> list[types.Part]:
  """Returns ``parts`` with adjacent text joined back together.

  Callers pass parts the transport split rather than the model: a stream gives
  each chunk of one text run its own part, and the aggregator ends a run at
  each resume.
  """
  joined: list[types.Part] = []
  for part in parts:
    _append_part(joined, part, at_pause=True)
  return joined


def _append_part(
    parts: list[types.Part], part: types.Part, *, at_pause: bool
) -> None:
  """Appends ``part``, joining it onto the text a pause split.

  A pause ends its text with an empty part carrying the thought signature, so
  that part joins onto the text it closes and the later signature wins. The
  streaming aggregator keeps those boundaries instead, which is why a resumed
  generation runs its result through here.

  Args:
    parts: The parts accumulated so far, appended to in place.
    part: The part to add.
    at_pause: Whether a pause separated ``part`` from the ones before it.
  """
  previous = parts[-1] if parts else None
  if previous is not None and _can_join(previous, part, at_pause=at_pause):
    parts[-1] = types.Part(
        text=(previous.text or '') + (part.text or ''),
        thought=previous.thought,
        thought_signature=part.thought_signature or previous.thought_signature,
    )
  elif part.text or part.thought_signature or not _only_has_text_fields(part):
    parts.append(part)


def _can_join(first: types.Part, second: types.Part, *, at_pause: bool) -> bool:
  """Whether ``second`` continues the text ``first`` started."""
  if not _only_has_text_fields(first) or not _only_has_text_fields(second):
    return False
  if not first.text:
    return False
  if not second.text:
    # A pause closes its text with a signature on an otherwise empty part.
    return bool(second.thought_signature)
  # Two text parts the model returned side by side are two parts, each with
  # its own signature; only a pause splits text that belongs together.
  return at_pause and bool(first.thought) == bool(second.thought)


def _only_has_text_fields(part: types.Part) -> bool:
  """Whether ``part`` carries nothing besides text, thought and signature."""
  return all(
      getattr(part, name) is None
      for name in type(part).model_fields
      if name not in _TEXT_FIELDS
  )


def _add_usage(
    total: types.GenerateContentResponseUsageMetadata | None,
    usage: types.GenerateContentResponseUsageMetadata | None,
) -> types.GenerateContentResponseUsageMetadata | None:
  """Returns the combined token usage of two requests."""
  if total is None:
    return usage
  if usage is None:
    return total
  combined: dict[str, object] = {}
  for name in type(usage).model_fields:
    if name == 'total_token_count':
      continue
    if name in _PROMPT_USAGE_FIELDS:
      first = getattr(total, name)
      combined[name] = first if first is not None else getattr(usage, name)
      continue
    values = [
        value
        for value in (getattr(total, name), getattr(usage, name))
        if value is not None
    ]
    if values and all(isinstance(value, int) for value in values):
      combined[name] = sum(values)
    elif values and all(isinstance(value, list) for value in values):
      combined[name] = _add_modality_counts(itertools.chain(*values))
  counts = [
      value
      for name in _TOTALLED_USAGE_FIELDS
      if isinstance(value := combined.get(name), int)
  ]
  if counts:
    combined['total_token_count'] = sum(counts)
  return usage.model_copy(update=combined)


def _add_modality_counts(
    counts: Iterable[types.ModalityTokenCount],
) -> list[types.ModalityTokenCount]:
  """Sums token counts that share a modality, keeping first-seen order."""
  totals: dict[types.MediaModality | None, int] = {}
  for count in counts:
    totals[count.modality] = totals.get(count.modality, 0) + (
        count.token_count or 0
    )
  return [
      types.ModalityTokenCount(modality=modality, token_count=token_count)
      for modality, token_count in totals.items()
  ]
