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

"""Hands the executor's session over to the advisor model.

What makes a consult worth more than a plain call to a larger model is that
the advisor sees what the executor saw: the same instructions, the same tool
results, the same dead ends. This module turns a session's event log into
content any advisor model can read, under a character budget so that a
long-horizon session cannot silently blow up the cost of a single consult.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ...events._rewind_events import _apply_rewinds

if TYPE_CHECKING:
  from ...events.event import Event

ContextMode = Literal['events', 'transcript']

_ROLE_LABELS = {'user': 'USER', 'model': 'AGENT'}
# How much more room plain text gets than a rendered tool payload. Documented
# on ModelConsultContextConfig.max_part_chars.
_TEXT_CHARS_MULTIPLIER = 8
_OMISSION_MARKER = (
    '[... {n} earlier turn(s) omitted to fit the context budget ...]'
)


class ModelConsultContextConfig(BaseModel):
  """Controls how much of the executor's session reaches the advisor."""

  model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

  mode: ContextMode = 'events'
  """How the session is shaped for the advisor.

  `'events'` hands over multi-turn `types.Content` objects; `'transcript'`
  collapses the session into one labelled plain-text block inside the final
  user message.
  """

  include_session: bool = True
  """Whether to send the session at all.

  When False, the advisor only sees the question and context that the executor
  passed as tool arguments.
  """

  max_events: int | None = Field(default=None, ge=1)
  """Keep at most this many of the most recent events. None keeps all.

  Counted over raw session events, before thoughts and other withheld parts
  are filtered out, so the advisor may end up seeing fewer turns than this.
  """

  max_chars: int | None = Field(default=200_000, ge=1)
  """Character budget for the handover.

  Whole turns are dropped from the middle once the budget is exceeded: the
  original task and the most recent turns are what the advisor needs. The
  newest turn is always kept, trimmed if it does not fit on its own.
  """

  max_part_chars: int = Field(default=4_000, ge=1)
  """Per-part cap on rendered tool calls and tool results.

  These are the usual source of runaway context. Plain model text gets
  `_TEXT_CHARS_MULTIPLIER` times this allowance, since prose is rarely what
  blows a session up and cutting an answer mid-sentence costs the advisor
  more than it saves.
  """

  include_media: bool = True
  """Whether to pass inline images and audio through to the advisor.

  Turn this off for text-only advisor models.
  """

  include_thoughts: bool = False
  """Whether to include the executor's own thought parts.

  Off by default: thought summaries are noisy, and they bias the advisor
  toward the framing the executor is already stuck in.
  """


def _truncate(text: str, limit: int) -> str:
  """Cuts `text` down to `limit` characters, noting how much was dropped.

  The note counts against the limit: a cap the caller set is a cap on what
  actually gets sent, not on what is left after the note is added.

  Args:
    text: The text to cut down.
    limit: The character budget for the returned string.

  Returns:
    The text, at most `limit` characters long.
  """
  if limit <= 0:
    return ''
  if len(text) <= limit:
    return text
  # Sized against the largest count that could be reported, so that the note
  # never pushes the result back over the limit.
  widest_note = f'\n[... {len(text)} characters truncated ...]'
  keep = limit - len(widest_note)
  if keep <= 0:
    return text[:limit]
  return f'{text[:keep]}\n[... {len(text) - keep} characters truncated ...]'


def _render_args(args: dict[str, Any] | None, limit: int) -> str:
  """Renders function call arguments as a JSON string."""
  if not args:
    return ''
  try:
    rendered = json.dumps(args, ensure_ascii=False, default=str)
  except (TypeError, ValueError):
    rendered = str(args)
  return _truncate(rendered, limit)


def _render_response(response: Any, limit: int) -> str:
  """Renders a function response body as a string."""
  if response is None:
    return ''
  if isinstance(response, str):
    rendered = response
  else:
    try:
      rendered = json.dumps(response, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
      rendered = str(response)
  return _truncate(rendered, limit)


def _convert_part(
    part: types.Part,
    config: ModelConsultContextConfig,
    skip_function_call_ids: frozenset[str],
) -> types.Part | None:
  """Normalizes one part into something any advisor model can read.

  Function calls and responses become readable text rather than live tool
  parts: the advisor does not hold the executor's tool declarations, and a
  dangling function call is a validation error for most providers.

  Args:
    part: The part to convert.
    config: The handover configuration.
    skip_function_call_ids: Function call ids to drop entirely.

  Returns:
    The converted part, or None when the part carries nothing worth sending.
  """
  if part.thought and not config.include_thoughts:
    return None

  if part.function_call is not None:
    call = part.function_call
    if call.id and call.id in skip_function_call_ids:
      return None
    args = _render_args(call.args, config.max_part_chars)
    return types.Part(text=f'[tool_call] {call.name}({args})')

  if part.function_response is not None:
    response = part.function_response
    if response.id and response.id in skip_function_call_ids:
      return None
    body = _render_response(response.response, config.max_part_chars)
    return types.Part(text=f'[tool_result] {response.name} -> {body}')

  if part.text is not None:
    text = _truncate(part.text, config.max_part_chars * _TEXT_CHARS_MULTIPLIER)
    if not text.strip():
      return None
    if part.thought:
      # Labelled, because rebuilding the part drops `thought` and the advisor
      # is being asked to doubt exactly this reasoning: it has to be able to
      # tell it apart from what the executor actually concluded.
      text = f'[thought] {text}'
    return types.Part(text=text)

  if part.inline_data is not None or part.file_data is not None:
    if config.include_media and config.mode != 'transcript':
      return part
    reason = '' if config.include_media else 'omitted'
    description = _describe_media_part(part, reason=reason)
    return None if description is None else types.Part(text=description)

  if part.executable_code is not None:
    code = _truncate(part.executable_code.code or '', config.max_part_chars)
    return types.Part(text=f'[code]\n{code}')

  if part.code_execution_result is not None:
    output = _truncate(
        part.code_execution_result.output or '', config.max_part_chars
    )
    return types.Part(text=f'[code_result] {output}')

  return None


def _part_chars(part: types.Part) -> int:
  """Estimates how much of the character budget one part consumes."""
  if part.text:
    return len(part.text)
  if part.inline_data is not None and part.inline_data.data:
    # Rough stand-in so that media still consumes budget.
    return len(part.inline_data.data) // 4
  return 0


def _content_chars(content: types.Content) -> int:
  """Estimates how much of the character budget one content consumes."""
  return sum(_part_chars(part) for part in content.parts or [])


def _merge_adjacent(contents: Sequence[types.Content]) -> list[types.Content]:
  """Collapses consecutive same-role contents into one.

  Gemini tolerates consecutive user turns, but several third-party advisor
  models reached through LiteLlm require strict role alternation, so the
  handover is normalized before it leaves.

  Args:
    contents: The contents to normalize, in order.

  Returns:
    The contents with adjacent same-role entries merged.
  """
  merged: list[types.Content] = []
  for content in contents:
    if merged and merged[-1].role == content.role:
      merged[-1] = types.Content(
          role=content.role,
          parts=list(merged[-1].parts or []) + list(content.parts or []),
      )
    else:
      merged.append(content)
  return merged


def _truncate_content(content: types.Content, limit: int) -> types.Content:
  """Trims a content down to `limit` characters.

  Media is charged against the budget on the same rough basis the budget was
  measured with, and replaced by a placeholder when it does not fit: a turn
  made of images would otherwise sail past the cap untouched.

  Args:
    content: The content to trim.
    limit: The character budget for this content.

  Returns:
    The content, trimmed to fit.
  """
  parts: list[types.Part] = []
  used = 0
  for part in content.parts or []:
    if used >= limit:
      continue
    if part.text is not None:
      parts.append(types.Part(text=_truncate(part.text, limit - used)))
      used += len(part.text)
      continue
    cost = _part_chars(part)
    if cost <= limit - used:
      parts.append(part)
      used += cost
      continue
    placeholder = _describe_media_part(
        part, reason='omitted to fit the context budget'
    )
    if placeholder is not None:
      placeholder = _truncate(placeholder, limit - used)
      if placeholder:
        parts.append(types.Part(text=placeholder))
        used += len(placeholder)
  return types.Content(role=content.role, parts=parts)


def _describe_media_part(part: types.Part, *, reason: str = '') -> str | None:
  """Names a non-text part in plain text, so its absence stays visible.

  One describer for every renderer: the transcript, the text-only conversion
  and the budget trim all name a part the same way, and a part kind that is
  handled here cannot be silently dropped by one of them.

  Args:
    part: The part to name.
    reason: Why the part is named instead of carried, when it was dropped.

  Returns:
    A bracketed description, or None when the part carries no media.
  """
  suffix = f' {reason}' if reason else ''
  if part.inline_data is not None:
    mime_type = part.inline_data.mime_type or 'unknown'
    return f'[media{suffix}: {mime_type}]'
  if part.file_data is not None:
    return f'[file{suffix}: {part.file_data.file_uri}]'
  return None


def _apply_char_budget(
    contents: list[types.Content], max_chars: int | None
) -> list[types.Content]:
  """Drops whole turns from the middle until the budget is met.

  Keeping the head preserves the original task; keeping the tail preserves the
  state the executor is actually stuck in. The newest turn is always kept, so
  when it alone is larger than the budget it is trimmed rather than allowed to
  undo the budget.

  Args:
    contents: The contents to trim, in order.
    max_chars: The character budget, or None for no budget.

  Returns:
    The contents, with an omission marker in place of any dropped turns.
  """
  if max_chars is None or not contents:
    return contents

  sizes = [_content_chars(content) for content in contents]
  if sum(sizes) <= max_chars:
    return contents

  head_budget = max_chars // 4
  head: list[types.Content] = []
  used = 0
  for content, size in zip(contents, sizes):
    if used + size > head_budget:
      break
    head.append(content)
    used += size

  # The marker is part of what gets sent, so it comes out of the budget too.
  remaining = max(max_chars - used - len(_OMISSION_MARKER), 0)
  tail: list[types.Content] = []
  tail_used = 0
  for content, size in zip(
      reversed(contents[len(head) :]), reversed(sizes[len(head) :])
  ):
    if tail_used + size > remaining and tail:
      break
    tail.append(content)
    tail_used += size
  tail.reverse()

  dropped = len(contents) - len(head) - len(tail)
  marker = (
      types.Content(
          role='user',
          parts=[types.Part(text=_OMISSION_MARKER.format(n=dropped))],
      )
      if dropped > 0
      else None
  )

  # The tail loop takes the newest turn whatever its size, so it is the one
  # place the budget can still be blown. Trim that turn instead of reporting a
  # cap the handover does not honour.
  newest = tail[-1]
  # Everything kept except the newest turn, which is what is left to trim.
  fixed = used + tail_used - _content_chars(newest)
  if marker is not None:
    if max_chars - fixed - _content_chars(marker) <= 0:
      # A budget this small cannot carry both. The newest turn is the state
      # the advisor is being asked about, so the marker is what goes.
      marker = None
    else:
      fixed += _content_chars(marker)

  kept = head + ([marker] if marker is not None else []) + tail
  allowance = max_chars - fixed
  if allowance < _content_chars(newest):
    kept = kept[:-1] + [_truncate_content(newest, max(allowance, 0))]

  # A turn that trimmed down to nothing is dropped rather than sent: a content
  # with no parts is a validation error for several providers, and it tells the
  # advisor nothing anyway.
  return [content for content in kept if content.parts]


def build_advisor_contents(
    events: Sequence[Event],
    *,
    config: ModelConsultContextConfig | None = None,
    skip_function_call_ids: Sequence[str] = (),
) -> list[types.Content]:
  """Converts session events into contents for the advisor request.

  Args:
    events: The session's event log, oldest first.
    config: The handover configuration. Defaults are used when omitted.
    skip_function_call_ids: Function call ids to drop, normally the in-flight
      consult itself, which the handoff message restates anyway.

  Returns:
    Normalized, budget-bounded contents. Empty when there is nothing to send.
  """
  config = config or ModelConsultContextConfig()
  if not config.include_session:
    return []

  skipped = frozenset(call_id for call_id in skip_function_call_ids if call_id)
  # Rewound invocations are still in the log but the executor no longer sees
  # them, and the point of the handover is that the advisor sees what the
  # executor saw. Same helper the prompt builder and the compactor use.
  live = _apply_rewinds(list(events))
  kept = [event for event in live if not event.partial]
  if config.max_events is not None:
    kept = kept[-config.max_events :]

  contents: list[types.Content] = []
  for event in kept:
    content = event.content
    if content is None or not content.parts:
      continue
    parts = [
        converted
        for part in content.parts
        if (converted := _convert_part(part, config, skipped)) is not None
    ]
    if not parts:
      continue
    author = event.author or content.role or 'model'
    role = 'user' if author == 'user' else 'model'
    contents.append(types.Content(role=role, parts=parts))

  # Merged twice on purpose: the budget pass can splice an omission marker
  # between two turns of the same role, which is exactly what the first merge
  # was there to rule out.
  trimmed = _apply_char_budget(_merge_adjacent(contents), config.max_chars)
  return _merge_adjacent(trimmed)


def render_transcript(contents: Sequence[types.Content]) -> str:
  """Renders contents as a labelled plain-text transcript.

  Args:
    contents: The contents to render, in order.

  Returns:
    The rendered transcript, with one labelled block per content.
  """
  lines: list[str] = []
  for content in contents:
    label = _ROLE_LABELS.get(content.role or 'model', 'AGENT')
    chunks: list[str] = []
    for part in content.parts or []:
      if part.text:
        chunks.append(part.text)
        continue
      description = _describe_media_part(part)
      if description is not None:
        chunks.append(description)
    if chunks:
      lines.append(f'{label}: ' + '\n'.join(chunks))
  return '\n\n'.join(lines)
