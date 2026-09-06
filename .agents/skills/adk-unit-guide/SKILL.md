---
name: adk-unit-guide
description: >-
  Writes a hands-on developer guide for one ADK code unit — a minimal runnable
  example, how it works, a configuration-option table, advanced uses,
  limitations, and links to related samples — to
  `docs/guides/{topic}/{unit}/index.md`, then lists it in the index at
  `docs/guides/README.md`. Its reader is a developer calling the unit from their
  own application, at more depth than the published adk.dev documentation
  carries. Use when asked to "write a unit guide for {class}", "document how to
  use {feature}", "add a guide for {file}", or after shipping a user-facing
  class, node, or plugin. Don't use for internals documentation aimed at someone
  changing or extending the unit — that is a design document under
  `docs/design/` (use `adk-unit-design`). Don't use to write a runnable sample
  under `contributing/samples/` (use `adk-sample-creator`).
---

# ADK code unit guide

A unit guide is granular usage documentation for one code unit, deeper than what
ships on adk.dev — so detail that would bloat the published documentation has
somewhere to live. The reader wants to call the unit from an application, so
lead with working code.

Unit guides focus on public APIs and caller-visible behavior. Do not
discuss internal implementation details (such as private methods, internal state
mechanisms, or unexported helpers).

## Voice

Write to help the reader understand, not to instruct them from above.

- **Give the reason, not only the rule.** Whenever the guide states a
  constraint, a default, or a recommendation, say why it is that way. A reader
  who knows the reason can handle the case the guide did not anticipate.
- **Do not decide for the reader.** Phrasings such as "most applications never
  need this" or "you will rarely" tell people what they want. State the
  trade-off and let them choose.
- **Explain at the caller's level.** Explaining what happens is required;
  explaining the machinery that makes it happen is not. If an explanation needs
  a private symbol to make sense, it is pitched at the wrong layer.
- **Length follows understanding.** Brevity is not the goal. Where a reader
  would have a follow-up question, answer it. Where the sentence already lands,
  leave it.
- **Problem before syntax.** Say what the reader is trying to do, then show the
  code.
- Every heading has at least one sentence under it before the next heading, and
  every code block has a sentence introducing what it does.

Sentence-level rules, following the Google developer documentation style guide:
present tense, no contractions, no parentheticals (use commas), no bare "This"
as a subject, no `e.g.` or `etc.`, no superlatives, sentence case in headings,
and no heading deeper than H3.

### What the difference looks like

From the `BaseNode` guide. Before:

> Most applications never subclass `BaseNode`, so the sections that follow cover
> the settings first.

That decides for the reader and gives them nothing to check their own case
against. After:

> You usually configure a node rather than subclass one, because the settings
> below cover what most graphs need. Subclassing earns its keep when you want
> behavior the settings cannot express, and that case is at the end.

Same length, same facts. The second one names the reason, so a reader can tell
which of the two situations is theirs.

A second pair, from the `JoinNode` guide. Before:

> Here three tasks run in parallel on the same input, and a `JoinNode` collects
> their results.

That opens in speech rather than documentation, and it drops the name of the
pattern the reader would search for. After:

> This example builds a fan-out/fan-in workflow. Three tasks run in parallel on
> the same input, and a `JoinNode` aggregates their results so that a final node
> can present all three together.

## Inputs

Require the source file, or a class or method named inside it. Also read, when
they exist: its unit tests (they give you an example to adapt) and its design
document at `docs/design/{topic}/{unit}/index.md`.

## Analyse before writing

- Purpose and intended use of the unit.
- Which classes depend on it, and which it depends on.
- Configuration options the unit itself introduces, ignoring inherited ones.
- Known limitations.
- Exclude internal implementation details such as private methods and
  attributes, internal helper functions, private execution state, or internal
  data structures.

## Where the guide goes

Mirror the source path under `docs/guides/`, one directory per unit, guide named
`index.md`. Drop the leading underscore of a private module:

| Source | Guide |
| :--- | :--- |
| `src/google/adk/workflow/_function_node.py` | `docs/guides/workflow/function_node/index.md` |
| `src/google/adk/plugins/reflect_retry_tool_plugin.py` | `docs/guides/plugins/reflect_retry_tool_plugin/index.md` |

Use named files instead of `index.md` only when one source file has genuinely
separate usage modes — `docs/guides/agents/llm_agent/` holds `single_turn.md`
and `task.md` for that reason.

Update an existing guide in place, keeping the existing wording wherever the
code has not changed, so the diff shows only what the change actually altered.

Then add the guide to `docs/guides/README.md` under the right category heading,
as `* [Title](path/index.md) - one-line summary.` That index is the only table
of contents; a guide missing from it is unreachable.

## Code examples

- One minimal example under "Get started", with enough of the surrounding
  classes to show where the call belongs. Start from a unit test if one exists.
- Keep the `google.adk` import lines, because the import path is the single
  most error-prone thing a reader copies. Omit unrelated standard-library
  imports and `asyncio.run(main())` runner boilerplate, which add nothing about
  the unit.
- Show what a developer would actually write. An example that only demonstrates
  the shape of an interface, or that drives a service the reader would normally
  reach through `Runner`, does not belong in a guide.
- Do not set `model=` on a sample agent — guides stay model-agnostic, and no
  guide in `docs/guides/` currently pins a model.
- For workflow nodes, show the logic as a plain Python function rather than a
  `BaseNode` subclass, unless the use case genuinely requires the subclass.
- Wrap a function as a node with the `@node` decorator rather than
  `FunctionNode` directly, except when demonstrating `FunctionNode`
  configuration itself.

## Link related samples

Link samples by repo-relative path from the guide, not by GitHub URL:
`[Node Output](../../../../contributing/samples/workflows/node_output/agent.py)`.
Confirm the file exists before linking it.

## Structure

Follow [references/guide-template.md](references/guide-template.md) section by
section.
