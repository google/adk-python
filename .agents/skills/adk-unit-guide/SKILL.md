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

## Inputs

Require the source file, or a class or method named inside it. Also read, when
they exist: its unit tests (they give you an example to adapt) and its design
document at `docs/design/{topic}/{unit}/index.md`.

## Analyse before writing

- Purpose and intended use of the unit.
- Which classes depend on it, and which it depends on.
- Configuration options the unit itself introduces, ignoring inherited ones.
- Known limitations.

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
