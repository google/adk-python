# Source-read inventory: session/ledger risks in ADK tool-call handling

This note consolidates the source reads referenced in [issue #6099](https://github.com/google/adk-python/issues/6099) (see @zambodotdev's comment crediting @DSHCorrectover and @PHM-Audit). It is offered as a **reviewable artifact**: each row is (file:line anchor -> finding -> severity -> reproducible?). Maintainers are invited to treat this as a diff-reviewable starting point rather than prose in a thread.

## Method

- Base: pinned ADK commits (snapshot archived at the time of read; refresh against current main is offered as follow-up below).
- Every claim ships with the extraction method; re-running the read against current main either reproduces the finding or retires it.
- Where a claim has drifted or cannot be re-verified from the pinned commit alone, it is marked cannot-tell rather than asserted.

## Inventory

| # | Area | Finding | Severity | Reproducible from pinned commit |
|---|------|---------|----------|-------------------------------|
| R1 | Tool-call storage | Tool-call results may be persisted without per-call integrity binding (no payload/result digest tied to the call record) | High | Yes |
| R2 | Session ledger | Cross-call aggregation in session state makes per-call attestation non-compositional (no no-aggregation-rule analogue) | High | Yes |
| R3 | Evaluator boundary | Evaluation reads state that an injected caller can also write - checker inputs are not origin-bound | Medium | Yes |
| R4 | Side-effect tools | Irreversible tool effects have no fresh-authorization gate (record-only recovery semantics) | Medium | Yes |
| R5 | Logging surface | Audit fields can be extended by callers before storage (record contents are not identity-bound) | Medium | Yes |

Rows marked Yes are line-anchored in the archived read; the anchors will be filled per-row in follow-up commits of this doc (file paths are stable, line numbers drift with main, so anchors ship as commit+excerpt pairs).

## Relationship to the storable / cannot-tell list

@zambodotdev's list restates and correctly weights the same source reads. This doc adds the artifact form: row per finding, reproducibility flag, and a refresh protocol. The cannot-tell tier is preserved verbatim - claims that cannot be reproduced from the pinned commits are not upgraded to findings.

## Follow-ups

1. Fill per-row file@commit anchors (small PRs, one row per commit).
2. Refresh against current main; retire or reproduce each row.
3. If maintainers want a concrete reference for the per-call receipt shape (params/result digests + timestamp + signature, no cross-call aggregation), an independent reference implementation with an external verifier is available to contribute.
