# Streaming preemption (`StreamingRouterNode`)

`StreamingRouterNode` advances a workflow graph **mid-stream**: it runs a wrapped
agent in SSE mode, watches the model's tokens as they arrive, and the moment a
caller-supplied monitor is confident, it commits a route/output and **cancels the
rest of the generation**. This turns "wait for the whole turn, then move on" into
"move on the instant the answer is knowable".

## Introduction

The stock LlmAgent-as-node wrapper only commits a node's output — the thing that
fires downstream triggers — on the *final*, non-partial event. The graph
therefore always advances at turn granularity: the model finishes generating,
then the scheduler moves on.

`StreamingRouterNode` closes that gap:

1. It forces the wrapped agent into `StreamingMode.SSE`.
2. It hands every streamed delta to a `monitor` callback.
3. When the monitor returns a `StreamDecision`, the node commits that decision's
   `route`/`output` and (by default) **closes the model stream**.

Closing the generator propagates `GeneratorExit` down ADK's `aclosing` chain,
which cancels the in-flight model call cooperatively — the same mechanism the
runtime already uses for node timeouts and interrupts. Because the node's
`run()` returns promptly, the scheduler advances the graph immediately instead
of paying for the tail of a turn the model has already effectively decided.

This is deterministic, mid-stream advancement: the graph moves as soon as the
decision is unambiguously present in the stream, and no wasted output tokens are
paid for.

## Get started

Fan out to several documents in parallel and, for each, stop generating the
moment a verdict has streamed in:

```python
from typing import Optional
from google.adk import Agent, Event, Workflow
from google.adk.workflow import (
    JoinNode,
    StreamDecision,
    StreamingRouterNode,
    StreamView,
)


def verdict_monitor(view: StreamView) -> Optional[StreamDecision]:
  """Preempt as soon as a `VERDICT:` line has fully streamed in."""
  idx = view.text.upper().find("VERDICT:")
  if idx == -1:
    return None
  line, sep, _ = view.text[idx + len("VERDICT:") :].partition("\n")
  if not sep:  # verdict line still streaming — keep reading
    return None
  verdict = line.strip()
  if verdict.upper().startswith("IRRELEVANT"):
    return StreamDecision(output={"relevant": False})
  if verdict.upper().startswith("RELEVANT"):
    return StreamDecision(output={"relevant": True, "verdict": verdict})
  return None


def make_reader(i: int, document: str) -> StreamingRouterNode:
  prompt = (
      f"PAPER:\n{document}\n\n---\n"
      "Is this a computer-science AI/ML paper?\n"
      "FIRST output a line 'VERDICT: RELEVANT - <topic>' or "
      "'VERDICT: IRRELEVANT'.\nTHEN write a long summary."
  )
  return StreamingRouterNode(
      name=f"reader_{i}",
      # A callable instruction bypasses {var} templating, so raw braces in the
      # document are sent verbatim.
      agent=Agent(name=f"reader_{i}", model="gemini-3.5-flash-lite",
                  instruction=lambda _ctx, _p=prompt: _p),
      monitor=verdict_monitor,
  )
```

Wire the readers into a fan-out/fan-in graph with a `JoinNode`; the scheduler
runs them concurrently and each one abandons its generation as soon as its
verdict lands.

## Benefits (measured)

The integration test
`tests/integration/test_streaming_router_preemption_timing.py` reads five whole
arXiv papers in parallel and asks "is this an AI paper?", comparing:

- **A** — read + answer, stream to completion.
- **B** — read + answer + SSE preemption (cut once the verdict streams in).

A representative run on real Vertex `gemini-3.5-flash-lite` (whole documents, no
chunking, ~199k shared input tokens):

| Metric | A (full) | B (preempt) |
| --- | --- | --- |
| Wall clock | 6.63s | 1.88s (**3.5x faster**) |
| Output tokens | 3,614 | 128 (**~28x fewer**) |
| Cost @ $0.30/$2.50 per 1M | $0.0686 | $0.0599 (**~13% cheaper**) |
| Cost w/ context caching (input @ $0.03/1M) | $0.0150 | $0.0063 (**~58% cheaper**) |

### Reading the numbers

- **Preemption saves *generation*, not *reading*.** Each document is one whole
  prompt in one call, so the model must prefill the entire input before it emits
  any token. Preemption cancels the *output* stream, which happens strictly after
  prefill — the input is already paid for. To also save reading you must not send
  the whole document in one call (incremental input), which is a different design.
- At standard pricing the per-query cost is **input-bound** (199k input vs a few
  thousand output tokens), so preemption's dollar impact is modest (~13%).
- With **context caching** the input read is 10x cheaper, output becomes the
  dominant cost, and preemption's ~28x output cut drives ~58% total savings.
  Context caching (amortize reading) and preemption (cut generation) are
  complementary.

## How it works

```python
async with Aclosing(self.agent.run_async(ic)) as run_iter:
  async for event in run_iter:
    if event.partial:
      # accumulate streamed text, hand it to the monitor
      decision = await self._invoke_monitor(StreamView(...))
      if decision is not None:
        self._apply_decision(ctx, decision)
        if decision.stop:
          return  # GeneratorExit -> aclosing cancels the model call
```

Key fields on `StreamingRouterNode`:

- `agent`: the (tool-free classifier/router) agent to stream.
- `monitor`: `Callable[[StreamView], Optional[StreamDecision]]`, sync or async.
- `forward_partials` (default `True`): re-yield partials as user-visible
  messages (typewriter effect) in addition to driving the monitor.
- `include_thoughts` (default `False`): include model `thought` parts in
  `StreamView.text`.

`StreamDecision(route=..., output=..., stop=True)` commits a routing value and/or
an output; `stop=True` (default) cancels the remaining generation, `stop=False`
lets it run to completion while the decision stands.

## Related: `enable_uvloop()`

For network-bound agent workloads you can install the libuv event loop with a
one-line switch at your entrypoint:

```python
import google.adk

google.adk.enable_uvloop()  # process-wide; call once before Runner.run
```

Deployments can opt in without touching code via `ADK_UVLOOP=1`, which the sync
`Runner.run` path honours. uvloop only accelerates code that actually awaits on
the loop (async client + `asyncio.gather`); it is the last 10%, not a 10x on its
own. Install with `pip install "google-adk[uvloop]"` (no Windows wheels).
```
