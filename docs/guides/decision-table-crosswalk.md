# Crosswalk: ADK tool-call authorization ↔ Correctover Recovery Admission decision table

This document proposes a mechanical mapping from the acceptance-harness sketch in
[#6099](https://github.com/google/adk-python/issues/6099) ("framework-emitted authority record" + "four independently testable parts") onto the 576-anchor decision table shipped in
[DSHCorrectover/correctover-recovery-admission](https://github.com/DSHCorrectover/correctover-recovery-admission)
(dataset CC0, code Apache-2.0, replayable with `python3 verify-recovery.py` in seconds, no install beyond Python 3.8+).

Rationale: #6099 asks for a closed, diffable set of authorization outcomes rather than prose.
The table is exactly that — the exhaustive cross-product of six dimensions — so an ADK fixture
can assert against expected verdicts instead of re-implementing semantics.

## Dimension → ADK field mapping

| Table dimension | Values | ADK framework-authority field (known at dispatch) |
|---|---|---|
| delegation shape | chained-valid · tail-mismatch · broken-chain · none-direct | principal/delegation chain resolved from session + tool context |
| revocation | none · revoked (chain propagates / local-only) | credential/consent revocation state |
| tool version | exact · patch · major | registered tool version vs approved baseline |
| side effect | readonly · state-write · failover · irreversible | tool metadata / declared effect class |
| runtime | fresh · stale · upstream-down | session/upstream freshness at dispatch |
| integrity | intact · tampered-digest | receipt digest verification |

## Verdict → expected outcome mapping

| Verdict | Reason code (stable identifier) | Fixture assertion |
|---|---|---|
| ALLOW | `clean` / `irreversible-authorized-exact-fresh` | dispatch proceeds; receipt carries class |
| DENY | `no-chain-cannot-bind-state-action`, `identity-revoked-propagated`, `identity-revoked-local`, … | call refused with reason code surfaced |
| ESCALATE | irreversible × (non-fresh or non-exact); failover × fresh-but-patch-drift | human gate required; gate block (`gate_id`, `gate_decision`, `gate_evidence_digest`, `bound_args_digest`) must re-bind exact dispatched args |

## The four acceptance parts, already measured

1. **Canonical bytes** — receipts bind `bound_args_digest`; same logical decision → same digest.
2. **Independent re-check** — verifier is a separate code path from the table; disagreement = FAIL.
3. **Negative branch live** — 288 intact anchors bind, 288 tampered anchors refuse (`verify-recovery.py --tamper` proves it).
4. **Recovery paths** — lost-response retry on mutating calls = `state-write × stale` → ESCALATE; a fresh JSON-RPC id without a dedup key fails the gate re-bind and is treated as a second action.

## Proposed use

- Emit the ADK framework-authority record at dispatch as today.
- Diff its fields against the closed verdict table (above mapping) for fixture expectations.
- Reason codes branch directly in hooks/tests; no verdict re-implementation.

All 576 anchors and expected verdicts are CC0 — the matrix can be vendored into ADK's test
fixtures without licensing concerns. If maintainers prefer a narrower start, the crosswalk can
begin with the `side-effect × runtime` sub-matrix (24 anchors) which covers the mutating-retry
class discussed in #6099.
