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

"""Mid-stream preemptive graph advancement.

The stock LlmAgent-as-node wrapper only commits a node's output — the thing
that fires downstream triggers — on the *final*, non-partial event. The graph
therefore always advances at turn granularity: the model finishes generating,
then the scheduler moves on.

``StreamingRouterNode`` closes that gap. It runs a wrapped agent in SSE mode
and hands every streamed delta to a caller-supplied ``monitor``. The moment the
monitor can decide — e.g. it has seen the routing token, a confident answer
prefix, or the classification JSON — it returns a :class:`StreamDecision`. The
node then:

  1. commits that decision's ``route`` / ``output`` to the context, and
  2. (by default) closes the model stream, cancelling the rest of the
     generation.

Closing the generator propagates ``GeneratorExit`` down ADK's ``aclosing``
chain, which cancels the in-flight model call cooperatively — the same
mechanism the runtime already uses for node timeouts and interrupts. Because
the node's ``run()`` then returns promptly, the workflow scheduler advances the
graph immediately instead of waiting for the tail of a turn the model has
already effectively decided.

This is deterministic, mid-stream advancement: the graph moves as soon as the
decision is unambiguously present in the stream, and no wasted tokens are paid
for. It intentionally does *not* speculatively dispatch a branch the model
might later revise; advancing only on a committed decision keeps the
correctness story simple. Keep the wrapped agent tool-free (a classifier /
router persona); function calls are streamed through but never trigger a
decision on their own.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..agents._streaming_mode import StreamingMode
from ..agents.llm_agent import LlmAgent
from ..agents.run_config import RunConfig
from ..events.event import Event
from ..utils.context_utils import Aclosing
from ._base_node import BaseNode

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._graph import RouteValue


@dataclass(frozen=True)
class StreamView:
  """A read-only snapshot of the model stream handed to the monitor.

  Attributes:
    text: All model text accumulated so far in this turn (thought parts
      excluded unless ``include_thoughts`` is set on the node).
    delta: The text carried by the current partial event (may be empty for
      non-text partials such as streaming function-call arguments).
    event: The raw partial :class:`Event` currently being processed.
  """

  text: str
  delta: str
  event: Event


class StreamDecision:
  """The monitor's verdict for advancing the graph mid-stream.

  Return an instance from a ``monitor`` to commit a routing decision and/or an
  output before the wrapped agent finishes its turn. Returning ``None`` means
  "not decided yet — keep streaming".

  At least one of ``route`` or ``output`` must be provided.

  Attributes:
    route: Routing value for conditional edges (a single value or a list).
      Read by the workflow scheduler to pick downstream edges.
    output: The node's output value. Also written to the agent's
      ``output_key`` (when set) as a state delta.
    stop: If True (default), the model stream is closed and the remaining
      generation cancelled the moment this decision is returned — the
      "preempt". If False, generation continues to completion while the
      decision stands (useful when you still want the full text persisted).
  """

  def __init__(
      self,
      *,
      route: Optional[Union[RouteValue, list[RouteValue]]] = None,
      output: Any = None,
      stop: bool = True,
  ) -> None:
    if route is None and output is None:
      raise ValueError(
          'StreamDecision requires at least one of `route` or `output`.'
      )
    self.route = route
    self.output = output
    self.stop = stop


# A monitor inspects each streamed delta and returns a decision (or None to
# keep waiting). It may be sync or async.
StreamMonitorCallback = Callable[
    [StreamView],
    Union[Optional[StreamDecision], Awaitable[Optional[StreamDecision]]],
]


def _model_text(event: Event, *, include_thoughts: bool) -> str:
  """Concatenates the model text on an event, skipping thoughts by default."""
  if not event.content or not event.content.parts:
    return ''
  return ''.join(
      part.text
      for part in event.content.parts
      if part.text and (include_thoughts or not part.thought)
  )


class StreamingRouterNode(BaseNode):
  """Advances the workflow graph mid-stream based on the model's own output.

  Wrap a classifier / router agent and supply a ``monitor`` predicate. The node
  streams the agent in SSE mode and, as soon as the monitor returns a
  :class:`StreamDecision`, commits the route/output and (by default) cancels the
  rest of the generation so the graph advances immediately.

  Example::

      def route_on_label(view: StreamView) -> StreamDecision | None:
        low = view.text.lower()
        if 'billing' in low:
          return StreamDecision(route='billing')
        if 'technical' in low:
          return StreamDecision(route='technical')
        return None

      router = StreamingRouterNode(
          name='intent_router',
          agent=LlmAgent(name='classifier', model='gemini-2.5-flash',
                         instruction='Reply with the single word intent.'),
          monitor=route_on_label,
      )
  """

  agent: LlmAgent = Field(...)
  """The agent to stream. Should be a tool-free classifier / router persona."""

  monitor: StreamMonitorCallback = Field(...)
  """Predicate called on every streamed delta; returns a decision or None."""

  forward_partials: bool = True
  """If True, streamed partial events are re-yielded as user-visible messages
  (typewriter effect) in addition to driving the monitor."""

  include_thoughts: bool = False
  """If True, model ``thought`` parts are included in ``StreamView.text``."""

  # Dynamic scheduling / resume support. Mirrors the LlmAgent-as-node default.
  rerun_on_resume: bool = True

  @model_validator(mode='after')
  def _validate_monitor(self) -> StreamingRouterNode:
    if not callable(self.monitor):
      raise ValueError('`monitor` must be callable.')
    return self

  def _apply_decision(self, ctx: Context, decision: StreamDecision) -> None:
    """Commits a decision's output and/or route onto the context."""
    if decision.output is not None:
      ctx.output = decision.output
      output_key = getattr(self.agent, 'output_key', None)
      if output_key:
        ctx.actions.state_delta[output_key] = decision.output
    if decision.route is not None:
      ctx.route = decision.route

  async def _invoke_monitor(self, view: StreamView) -> Optional[StreamDecision]:
    result = self.monitor(view)
    if inspect.isawaitable(result):
      return await result
    return result

  def _build_streaming_ic(self, ctx: Context, node_input: Any) -> Any:
    """Prepares an InvocationContext that streams the wrapped agent via SSE."""
    from ._llm_agent_wrapper import prepare_llm_agent_context
    from ._llm_agent_wrapper import prepare_llm_agent_input

    agent = self.agent
    if agent.mode is None:
      agent.mode = 'single_turn'
    # As a single-turn node, default to not replaying prior turns unless the
    # author opted in — matching run_llm_agent_as_node.
    if (
        agent.mode == 'single_turn'
        and 'include_contents' not in agent.model_fields_set
    ):
      agent.include_contents = 'none'

    agent_ctx = prepare_llm_agent_context(agent, ctx)
    prepare_llm_agent_input(agent, agent_ctx, node_input)

    ic = agent_ctx.get_invocation_context()
    run_config = (ic.run_config or RunConfig()).model_copy(
        update={'streaming_mode': StreamingMode.SSE}
    )
    update: dict[str, Any] = {'agent': agent, 'run_config': run_config}
    iso = getattr(agent_ctx, 'isolation_scope', None)
    if agent.mode in ('task', 'single_turn') and iso:
      update['isolation_scope'] = iso
    return ic.model_copy(update=update)

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ic = self._build_streaming_ic(ctx, node_input)

    decided = False
    accumulated: list[str] = []

    async with Aclosing(self.agent.run_async(ic)) as run_iter:
      async for event in run_iter:
        if event.partial:
          delta = _model_text(event, include_thoughts=self.include_thoughts)
          if delta:
            accumulated.append(delta)
          if self.forward_partials:
            yield event
          if not decided:
            decision = await self._invoke_monitor(
                StreamView(text=''.join(accumulated), delta=delta, event=event)
            )
            if decision is not None:
              self._apply_decision(ctx, decision)
              decided = True
              if decision.stop:
                # Returning closes the stream (GeneratorExit -> aclosing),
                # cancelling the rest of the model call, and lets the
                # scheduler advance on the committed route/output.
                return
          continue

        # Non-partial (aggregated / final) event.
        if decided:
          # A decision already owns this node's output/route. Stream the
          # event for visibility but strip any output it carries to avoid a
          # double-set on the context.
          if event.output is not None:
            event = event.model_copy(update={'output': None})
          yield event
          continue

        # No early decision: fall back to standard output extraction.
        from ._llm_agent_wrapper import process_llm_agent_output

        process_llm_agent_output(self.agent, ctx, event)
        yield event
