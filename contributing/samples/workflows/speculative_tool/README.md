# Speculative Tool Dispatch

## Overview

Don't wait for the model to finish asking — start the work the moment you can
guess what it's asking for, then verify.

`SpeculativeRouterNode` is the aggressive counterpart to `StreamingRouterNode`:

| | `StreamingRouterNode` | `SpeculativeRouterNode` |
|---|---|---|
| Strategy | **conservative** — wait for a committed decision, then cancel the tail | **aggressive** — act on a *partial* call, then verify |
| On the args | doesn't touch them | **repairs** truncated JSON and dispatches early |
| Risk | ~none | can mis-guess → cancel + re-run (must be idempotent) |

## How it works

The model emits a directive line:

```
TOOL_CALL: {"name": "read_file", "arguments": {"path": "src/main.c"}}
```

As it streams, the node:

1. **Extracts + repairs.** The default extractor finds `TOOL_CALL:`, takes the
   JSON after it, and — if it's still truncated (`{"path": "src/ma`) — runs it
   through `repair_json` to get a parseable, best-effort payload.
2. **Dispatches early.** The first payload that passes `should_speculate` is used
   to run the `target` node **immediately**, overlapping with generation.
3. **Verifies.** When the finalized call arrives it compares (via `same`):
   - **hit** → keep the speculative result (already done or nearly so);
   - **miss** → cancel the speculative run (cooperatively tearing down its
     in-flight work) and re-run the target with the correct payload.

The node's output is the *verified* target output.

## Safety: idempotent targets only

Speculation means the target may run on a wrong guess and be cancelled. Only use
it for **read-only / idempotent** work — file/DB reads, search, retrieval. Never
speculate a side-effecting action (sending mail, writing files, charging a card).

Knobs:

- `should_speculate(payload) -> bool` — gate early dispatch (e.g. only once a
  path is long enough to be worth guessing).
- `same(a, b) -> bool` — how hit/miss is decided (here: same resolved `path`).
- `extract(text) -> payload | None` — plug in a different protocol, or add a
  parameter-prediction step (e.g. complete a partial path against the repo).
- `timeout` — bounds the speculative read so a bad guess can't hang the turn.

## Provenance

This mirrors a libuv-based agent runtime that repairs partial tool-call JSON and
fires the call before the stream closes, then reconciles at end-of-stream — ported
to ADK as a first-class, cancellable graph node built on `ctx.run_node`.
```
