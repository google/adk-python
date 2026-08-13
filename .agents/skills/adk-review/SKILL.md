---
name: adk-review
description: >-
  Reviews the uncommitted changes in an adk-python working tree and reports
  correctness, design, public-API stability, test, sample and documentation
  gaps as a prioritized findings report, fixing them only when asked. Use when
  the user asks to review local changes, wants a self-review before opening a
  pull request, asks whether a change breaks the public API or needs tests,
  samples or docs, or asks what is wrong with the current diff. Required for
  changes to public APIs, core architecture (Runner, Workflow, BaseNode), new
  features and major refactors. Don't use for a single style nit (use
  adk-style), for diagnosing a failing test or a misbehaving agent at runtime
  (use adk-debug), or for wording a commit message or PR description (use
  adk-git).
---

# ADK Change Reviewer

Review the working-tree diff against the seven dimensions below, report what is
wrong, and stop. Fix only what the user then asks you to fix.

## Workflow

1. `git status` and `git diff` (add `--staged` for staged work) to get the exact
   set of added, modified, and deleted files.
2. Review the diff file by file against the checklist.
3. Emit the report in the format below.
4. Stop. Do not edit any file, and do not offer to. Wait for the user to ask.
5. If the user asks for fixes, apply them, then re-run the affected tests
   (`pytest tests/unittests/{path}`) and `pre-commit run --files {paths}`.

Step 4 is the part that is easy to get wrong: an unrequested fix buries the
findings the user asked for and mixes review output with new, unreviewed edits.

## Checklist

### 1. Correctness

- **Types**: no new `mypy` errors. CI diffs `mypy` output against the base
  branch and fails only on *newly introduced* errors, so a pre-existing error in
  a file you touched is not a blocker but a new one is.
- **Imports**: no circular imports; absolute imports where the module already
  uses them.
- **Exceptions**: no bare `except:`; catch a specific type and log with enough
  context to identify the caller.
- **Type discrimination**: check an object's type before reading a
  type-specific attribute (for example confirm a node is an `LlmAgent` before
  inspecting `mode`), so an unexpected node type raises nothing.
- **Boundaries**: `None`, empty collections, zero, and empty strings are
  handled by validation or a fallback default.
- **Preconditions**: state invariants are checked before the core logic runs.

### 2. Design

- **Complexity**: functions or classes that would be clearer split up.
- **Coupling**: high cohesion, low coupling; no anti-patterns introduced.
- **Performance**: redundant computation, repeated I/O in a loop, or
  allocations that scale with input where they need not.
- **Security**: inputs validated, sensitive data not logged, no injection,
  resource exhaustion, or exposure of internal state.

### 3. Style

Cross-reference the diff against the `adk-style` skill rather than restating
its rules here: visibility and `_` prefixes, typing, Pydantic v2 patterns, lazy
logging, imports, async, and file organization.

Confirm the changed files pass `pre-commit run --files {paths}`.

### 4. Architecture and unintended outcomes

- **Public API stability**: does the change modify, remove, or narrow a
  public interface, class, method, argument list, or CLI surface under
  `src/google/adk/`? A breaking change needs a deprecation cycle first, because
  the package is released under Semantic Versioning and users pin minor
  versions.
- **Execution and resumption**: changes to workflows, nodes, or state must stay
  compatible with the event execution lifecycle and with session resumption
  (human-in-the-loop steps and checkpoints). See the `adk-architecture` skill.
- **Concurrency and lifetime**: no race conditions; plugins, exporters, and
  connections are closed on every path, including the error path.

### 5. Documentation impact

- User-facing documentation lives in the separate `adk-docs` repository, so a
  user-visible change needs a PR there as well; note it in the report.
- Guides under `docs/guides/` may need an update when a public API or workflow
  pattern changes.
- If the change alters a code unit's design contract, the `adk-unit-design`
  skill owns the design document for that unit.

### 6. Samples

- Do existing samples under `contributing/samples/` still run against the
  change?
- Does a new capability warrant a new sample? Follow the `adk-sample-creator`
  conventions if so.

### 7. Tests

- Every new or modified code path has a test under `tests/unittests/`.
- A new test belongs in the existing `test_{module}*.py` file for its unit, not
  in a file named after the change, which fragments that unit's coverage.
- Tests follow the rules in the `adk-style` testing reference: one behavior per
  test, behavior-named tests, no assertions on private attributes, minimal
  fixtures, arrange/act/assert structure.

## Report format

Group findings by priority and give a file path and line for each. Do not
report a dimension with nothing to say.

- 🔴 **Critical**: bugs, type errors, race conditions, resource leaks, security
  issues.
- 🟠 **Design**: complexity, readability, performance, architectural
  misalignment.
- 🟡 **Style**: lint, formatting, non-lazy logging, typing mismatches.
- 🔵 **Docs, tests, samples**: missing or stale coverage, guides, samples.
