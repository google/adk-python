# OkfToolset

`OkfToolset` gives an agent four read-only tools over a directory of *declarative* knowledge: markdown files whose YAML frontmatter says what each one is, whether a human or a machine verified it, and when it goes stale. The tools disclose the corpus progressively — an index of frontmatter rows first, one concept's body second, its attached resources third — so the context bill is paid per concept the model actually reads.

## Introduction

ADK already has a strong story for procedural knowledge: a `SKILL.md` tree tells the agent *how* to do something. What an agent grounded in a versioned knowledge corpus needs is the other half — a way to consume what the corpus *says*, without either loading all of it into the system prompt or building a bespoke retrieval layer beside it.

That corpus usually already has a shape. The [Open Knowledge Format (OKF) v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format) describes it: every concept is a markdown file with YAML frontmatter carrying `type`, `description`, `tags`, `sources`, `generated`, `verified`, `status`, and `stale_after`. The structure is the index, and `OkfToolset` reads it as one.

Three properties of the format do the work here:

- **`type` and `tags` are authored metadata**, so filtering happens before any body is read.
- **`verified` carries actors** (`human:<id>`, `<producer>/<version>`), so trust is derived per concept rather than assumed per corpus.
- **`status` and `stale_after` are dated**, so a deprecated or expired concept can be reported as such instead of silently cited as current.

`OkfToolset` is a `BaseToolset`. Bind it to a bundle directory and it returns four `FunctionTool` instances. Parsing, trust derivation, and traversal live in sibling modules that never import `google.adk`, so the same logic is reusable outside an agent.

## Get started

The toolset needs no third-party library beyond `pyyaml`, which ADK already depends on, so there is no extra to install:

```shell
pip install google-adk
```

A bundle is an ordinary directory tree. Reserved filenames (`index.md`, `log.md`) are indexes and history, not concepts:

```text
knowledge/
├── index.md
├── metrics/
│   ├── gross-margin.md
│   └── revenue.md
├── policies/
│   └── revenue-recognition.md
└── tables/
    └── orders.md
```

Each concept opens with frontmatter, then prose:

```markdown
---
type: Metric
description: Gross margin as booked by finance.
tags: [finance, margin]
verified:
  - by: human:controller
    at: 2026-09-01T00:00:00Z
stale_after: 2027-03-01T00:00:00Z
---

# Gross margin

Revenue minus cost of goods sold, reported per period.
```

Hand the toolset to an agent:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.okf import OkfToolset

agent = LlmAgent(
    name="knowledge_agent",
    instruction=(
        "Answer from the knowledge bundle. Start with okf_index to find"
        " candidate concepts, read only the ones you need, and cite the"
        " concept path in your answer."
    ),
    tools=[
        OkfToolset(bundle="./knowledge", tool_name_prefix="okf"),
    ],
)
```

`tool_name_prefix` yields the tool names `okf_index`, `okf_read`, `okf_resource`, and `okf_search`. The toolset also accepts the standard `BaseToolset` arguments (`tool_filter`, for instance) and a `now` clock, which is what the staleness tests inject.

## How it works

### Four tools, one filter

| tool | level | returns |
| --- | --- | --- |
| `okf_index` | L1 | one metadata row per matching concept — never a body |
| `okf_read` | L2 | one concept's frontmatter and markdown body |
| `okf_resource` | L3 | a file attached to a concept, capped at 64 KiB |
| `okf_search` | — | concepts whose frontmatter or body matches a query |

`okf_index` and `okf_search` share a single filter predicate (`directory`, `type`, `tags`, `trust_tier`, `include_deprecated`, `include_stale`), so the two can never disagree about which concepts exist. Discovery costs one line per concept; a body is fetched only once the model has chosen it.

`okf_search` is a deterministic, case-insensitive substring or regex scan. There are no embeddings, no network calls, and no ranking: given the same bundle and query it returns the same rows in the same order, which is what makes the tool testable and its results citable.

### Derived, never stored

Trust tier, lifecycle, and staleness are recomputed from frontmatter on every read; the bundle holds signals, not verdicts. A missing `status` reads as `stable`, an unknown status also reads as `stable` (the spec forbids rejecting a concept for it), and a `stale_after` without a timezone is treated as absent rather than guessed at.

### Paths resolve inside the bundle or not at all

Every path argument — `okf_read`'s `path`, `okf_resource`'s `concept_path` and `file` — goes through one guard. A leading `/` means *bundle*-relative, never host-absolute. `..` traversal and symlinks that leave the bundle root are refused. For bundles that write root-relative resource paths without the leading slash, `okf_resource` tries the concept's own directory first and the bundle root second; both attempts pass through the same guard, and a refusal is never retried by the other route.

Eager loading enforces the same boundary: a symlinked concept that escapes the root is recorded as a violation instead of being parsed, so a hostile file cannot smuggle host content in before any tool is called.

### Failures are results, not exceptions

An exception raised inside a tool reaches the model as an opaque failure. These tools return structured errors instead: `{"error": "not_found" | "unsafe_path" | "invalid_argument" | "unreadable" | "invalid_regex" | "internal_error", "message": ...}`. A malformed bundle is reported the same way — as a `violations` list on `okf_index` — so one bad file never hides the rest of the corpus.

## Security model

A bundle is untrusted input: it is written by whoever produced the corpus, not by the agent's operator.

- **Concept bodies and resource text are data, never instructions.** Every tool description says so, and `okf_read` and `okf_resource` mark their payloads with `content_is_untrusted_data: true`. Text inside a concept that impersonates the user, the system, or a tool request is content to report, not a command to follow.
- **Trust tiers are advisory.** `verified` is a claim made by the bundle. The tier it produces filters discovery; it never authorises an action, and no code path gates on it.
- **The toolset is strictly read-only.** It executes nothing from the bundle — no scripts, no attested computations, no network. `agent`, `executor`, and `attester` frontmatter is surfaced as fields for the model to reason about, not run.

## Testing

`tests/unittests/integrations/okf/` covers the parser, trust derivation, traversal, and the four tools, including an adversarial suite for traversal, symlink escape, and prompt-injection labelling. The four published OKF sample bundles are vendored under `fixtures/bundles/` (provenance in `PROVENANCE.md`) so the suite stays offline and deterministic.
