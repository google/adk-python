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

"""Speculative mid-stream dispatch: act on a partial call, verify, roll back.

Where :class:`StreamingRouterNode` is *conservative* — it waits for a committed
decision in the stream and then cancels the tail — ``SpeculativeRouterNode`` is
*aggressive*. As the model streams a structured call (e.g. a ``TOOL_CALL: {...}``
directive), this node repairs the still-incomplete JSON, dispatches a downstream
target node **immediately** with that best-effort payload, and lets the model
keep generating. When the finalized call arrives it verifies:

  * **hit**  — the finalized payload matches the speculated one → keep the
    speculative result (its work overlapped generation, so it's already done or
    nearly so), and
  * **miss** — they differ → cancel the speculative run (cooperatively tearing
    down its in-flight work) and re-dispatch the target with the correct
    payload.

This trades peak latency for speculation risk: a wrong guess wastes a run and
must be safe to cancel. Use it only for **idempotent / side-effect-free**
targets (reads, searches, retrieval) — never for a target that, say, sends an
email or charges a card. This is the ADK analogue of a libuv agent runtime that
fires repaired tool calls before the stream closes.

The default extractor looks for a marker (``TOOL_CALL:`` by default), takes the
JSON that follows, and — if it is truncated — runs it through :func:`repair_json`
before parsing. Supply your own ``extract`` to plug in a different protocol or a
parameter-prediction step (e.g. completing a partial file path).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable
import json
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.llm_agent import LlmAgent
from ..events.event import Event
from ..utils.context_utils import Aclosing
from ._base_node import BaseNode
from ._graph import NodeLike
from ._retry_config import RetryConfig
from ._streaming_router import _model_text
from ._streaming_router import build_sse_invocation_context
from .utils._workflow_graph_utils import build_node

if TYPE_CHECKING:
  from ..agents.context import Context

logger = logging.getLogger('google_adk.' + __name__)

_UNSET = object()


def repair_json(fragment: str) -> str:
  """Completes a truncated JSON fragment into a parseable string (best effort).

  A stack-based state machine (ported from the ``syncrig`` C repairer): it closes
  an open string, finishes a truncated ``true``/``false``/``null`` literal, drops
  a dangling comma, fills a dangling key/colon with ``null``, and appends the
  ``}``/``]`` needed to balance every open object/array. Given
  ``'{"path": "src/ma'`` it returns ``'{"path": "src/ma"}'``.
  """
  _OBJ, _ARR = 0, 1
  out: list[str] = []
  stack: list[int] = []
  state: list[int] = []  # per-frame object state: 0 key,1 colon,2 value,3 comma
  in_string = False
  escaped = False

  for c in fragment:
    out.append(c)
    if in_string:
      if escaped:
        escaped = False
      elif c == '\\':
        escaped = True
      elif c == '"':
        in_string = False
        if stack and stack[-1] == _OBJ:
          if state[-1] == 0:
            state[-1] = 1
          elif state[-1] == 2:
            state[-1] = 3
      continue
    if c == '"':
      in_string = True
    elif c == '{':
      stack.append(_OBJ)
      state.append(0)
    elif c == '[':
      stack.append(_ARR)
      state.append(-1)
    elif c == '}':
      if stack and stack[-1] == _OBJ:
        stack.pop()
        state.pop()
        if stack and stack[-1] == _OBJ and state[-1] == 2:
          state[-1] = 3
    elif c == ']':
      if stack and stack[-1] == _ARR:
        stack.pop()
        state.pop()
        if stack and stack[-1] == _OBJ and state[-1] == 2:
          state[-1] = 3
    elif c == ':':
      if stack and stack[-1] == _OBJ:
        state[-1] = 2
    elif c == ',':
      if stack and stack[-1] == _OBJ:
        state[-1] = 0

  if in_string:
    if escaped:
      out.pop()  # dangling backslash
    out.append('"')
    if stack and stack[-1] == _OBJ:
      if state[-1] == 0:
        state[-1] = 1
      elif state[-1] == 2:
        state[-1] = 3

  while out and out[-1] in ' \n\r\t':
    out.pop()

  if out and out[-1] == ',':
    out.pop()
    while out and out[-1] in ' \n\r\t':
      out.pop()
    if stack and stack[-1] == _OBJ:
      state[-1] = 3

  if out and (out[-1].isalnum() or out[-1] in '.-'):
    s = ''.join(out)
    j = len(s)
    while j > 0 and s[j - 1].isalpha():
      j -= 1
    frag = s[j:]
    completed = False
    for kw in ('true', 'false', 'null'):
      if frag and frag != kw and kw.startswith(frag):
        out.extend(kw[len(frag) :])
        completed = True
        break
    if stack and stack[-1] == _OBJ and state[-1] == 2:
      state[-1] = 3
    del completed

  if stack and stack[-1] == _OBJ:
    if state[-1] == 1:
      out.append(':null')
      state[-1] = 3
    elif state[-1] == 2:
      out.append('null')
      state[-1] = 3

  while stack:
    out.append('}' if stack.pop() == _OBJ else ']')

  return ''.join(out)


def make_marker_extractor(marker: str = 'TOOL_CALL:') -> Callable[[str], Any]:
  """Builds an extractor that pulls the (possibly-partial) JSON after ``marker``.

  Returns the parsed object, or ``None`` if the marker/JSON has not appeared yet.
  A complete object is parsed as-is (trailing text ignored); a truncated one is
  run through :func:`repair_json` first.
  """

  def extract(text: str) -> Any:
    idx = text.find(marker)
    if idx == -1:
      return None
    rest = text[idx + len(marker) :]
    brace = rest.find('{')
    if brace == -1:
      return None
    fragment = rest[brace:]
    try:
      obj, _ = json.JSONDecoder().raw_decode(fragment)
      return obj
    except json.JSONDecodeError:
      pass
    try:
      return json.loads(repair_json(fragment))
    except json.JSONDecodeError:
      return None

  return extract


class SpeculativeRouterNode(BaseNode):
  """Speculatively dispatches a target node from a partial streamed call.

  Wrap a streaming agent and a ``target`` node. As the agent streams, ``extract``
  turns the accumulated text into a payload (repairing truncated JSON); the first
  time it yields something ``should_speculate`` accepts, the target is dispatched
  with that payload while generation continues. When the finalized payload
  arrives it is compared with ``same``: on a match the speculative result is kept;
  otherwise the speculative run is cancelled and the target is re-run with the
  finalized payload. The node's output is the target's (verified) output.

  If ``combine`` is supplied, the node's output is instead
  ``combine(agent_full_text, target_output)`` — this makes the agent's complete
  streamed text (e.g. a planner's rationale) a *required* returned deliverable
  rather than a discarded tail, so a non-speculative baseline that must also emit
  that text gets no "just plan and stop early" shortcut.

  The target must be safe to run speculatively and to cancel — use read-only /
  idempotent work only.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  agent: LlmAgent = Field(...)
  """The streaming agent whose output is parsed for a call to dispatch."""

  marker: str = 'TOOL_CALL:'
  """Marker the default extractor searches for before the JSON payload."""

  forward_partials: bool = False
  """If True, re-yield the agent's partial events as user-visible messages."""

  include_thoughts: bool = False
  """If True, model ``thought`` parts are included in the accumulated text."""

  emit_speculation_events: bool = True
  """If True, yield lightweight hit/miss/dispatch events for observability."""

  rerun_on_resume: bool = True

  _target: BaseNode = PrivateAttr()
  _extract: Callable[[str], Any] = PrivateAttr()
  _should_speculate: Callable[[Any], bool] = PrivateAttr()
  _same: Callable[[Any, Any], bool] = PrivateAttr()
  _combine: Optional[Callable[[str, Any], Any]] = PrivateAttr(default=None)

  def __init__(
      self,
      *,
      name: str,
      agent: LlmAgent,
      target: NodeLike,
      extract: Optional[Callable[[str], Any]] = None,
      should_speculate: Optional[Callable[[Any], bool]] = None,
      same: Optional[Callable[[Any, Any], bool]] = None,
      combine: Optional[Callable[[str, Any], Any]] = None,
      marker: str = 'TOOL_CALL:',
      forward_partials: bool = False,
      include_thoughts: bool = False,
      emit_speculation_events: bool = True,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    super().__init__(
        name=name,
        agent=agent,
        marker=marker,
        forward_partials=forward_partials,
        include_thoughts=include_thoughts,
        emit_speculation_events=emit_speculation_events,
        retry_config=retry_config,
        timeout=timeout,
    )
    self._target = build_node(target)
    self._extract = extract or make_marker_extractor(marker)
    self._should_speculate = should_speculate or (lambda _payload: True)
    self._same = same or (lambda a, b: a == b)
    self._combine = combine

  async def _run_target(self, ctx: Context, payload: Any) -> Any:
    return await ctx.run_node(
        self._target, node_input=payload, use_sub_branch=True
    )

  def _info(self, kind: str, payload: Any) -> Event:
    return Event(author=self.name, message=f'[speculation:{kind}] {payload}')

  def _commit(self, ctx: Context, result: Any, plan_text: str = '') -> None:
    # When ``combine`` is set the node's deliverable is a function of *both* the
    # agent's full text (e.g. a planner's rationale that the caller requires) and
    # the target's result — so the streamed text is a returned artifact, not a
    # throwaway tail that only exists to make speculation look good.
    out = self._combine(plan_text, result) if self._combine else result
    ctx.output = out
    output_key = getattr(self.agent, 'output_key', None)
    if output_key and out is not None:
      ctx.actions.state_delta[output_key] = out

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ic = build_sse_invocation_context(self.agent, ctx, node_input)

    accumulated: list[str] = []
    final_text = ''
    spec_task: Optional[asyncio.Task[Any]] = None
    spec_payload: Any = _UNSET
    all_tasks: list[asyncio.Task[Any]] = []

    try:
      async with Aclosing(self.agent.run_async(ic)) as run_iter:
        async for event in run_iter:
          if event.partial:
            delta = _model_text(event, include_thoughts=self.include_thoughts)
            if delta:
              accumulated.append(delta)
            if self.forward_partials:
              yield event
            payload = self._extract(''.join(accumulated))
            if (
                payload is not None
                and self._should_speculate(payload)
                and not (
                    spec_payload is not _UNSET
                    and self._same(payload, spec_payload)
                )
            ):
              # (Re)dispatch: the stream refined the payload, so abandon the
              # prior guess and speculate on the newer one.
              if spec_task is not None and not spec_task.done():
                spec_task.cancel()
              spec_payload = payload
              spec_task = asyncio.create_task(self._run_target(ctx, payload))
              all_tasks.append(spec_task)
              if self.emit_speculation_events:
                yield self._info('dispatch', payload)
            continue

          # Non-partial (aggregated / final) event: capture the full text.
          text = _model_text(event, include_thoughts=self.include_thoughts)
          if text:
            final_text = text

      plan_text = final_text or ''.join(accumulated)
      final_payload = self._extract(plan_text)

      if final_payload is None:
        # Nothing actionable ever materialized; drop any speculation.
        return

      if spec_task is not None and self._same(final_payload, spec_payload):
        # HIT: the speculated payload was right — keep its (overlapped) result.
        try:
          result = await spec_task
        except asyncio.CancelledError:
          raise
        except Exception as e:  # pylint: disable=broad-except
          logger.warning(
              'SpeculativeRouterNode %s: speculative run failed (%s);'
              ' re-running.',
              self.name,
              e,
          )
          result = await self._run_target(ctx, final_payload)
        else:
          if self.emit_speculation_events:
            yield self._info('hit', final_payload)
        self._commit(ctx, result, plan_text)
        return

      # MISS (or never speculated): cancel the wrong guess, run the real one.
      if spec_task is not None:
        spec_task.cancel()
        await asyncio.gather(spec_task, return_exceptions=True)
        if self.emit_speculation_events:
          yield self._info('miss', final_payload)
      result = await self._run_target(ctx, final_payload)
      self._commit(ctx, result, plan_text)
    finally:
      leftovers = [t for t in all_tasks if not t.done()]
      for t in leftovers:
        t.cancel()
      if leftovers:
        await asyncio.gather(*leftovers, return_exceptions=True)
