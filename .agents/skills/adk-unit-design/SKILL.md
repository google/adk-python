---
name: adk-unit-design
description: >-
  Writes an as-built architecture document for one ADK code unit — purpose,
  execution flow, data flow, cross-class dependencies, extension points, and
  the parts that must not change — to `docs/design/{topic}/{unit}/index.md`. It
  describes the code as implemented, not a proposed design, and its reader is a
  developer about to change or extend that unit. Use when asked to "write a
  design doc for {file}", "document the architecture of {class}", "document the
  extension points of {unit}", or after adding a core class, node type, or
  plugin base. Don't use for documentation aimed at developers who only call the
  unit from their own application — that is a usage guide with runnable examples
  under `docs/guides/` (use `adk-unit-guide`). Don't use to answer a
  framework-wide architecture question (use `adk-architecture`).
---

# ADK code unit design

A unit design documents a code unit **as implemented**, the way a unit test
exercises it as implemented. Nothing proposed or aspirational belongs in one.
The reader is deciding what they may safely change, so answer "what breaks if I
touch this?" — not "how do I call this?".

## Inputs

Require the source file, or a class or method named inside it. Also read, when
they exist: the base classes and interfaces the unit implements, its unit tests
(the best evidence of intended behaviour), and example usage.

## Analyse before writing

Answer each of these from the source. Anything the code does not show stays out
of the document — do not infer a design intent from a name.

- Purpose and intended use of the unit.
- Execution flow, and the data that flows in and out.
- Upstream dependencies, and which classes depend on this unit.
- Extension surfaces: abstract methods, hooks, callbacks, configurable fields.
- Constraints — what a subclass or caller must not change, and why.
- Operational limitations.

## Where the document goes

Mirror the source path under `docs/design/`, one directory per unit, document
named `index.md`. Drop the leading underscore of a private module:

| Source | Design document |
| :--- | :--- |
| `src/google/adk/workflow/_function_node.py` | `docs/design/workflow/function_node/index.md` |
| `src/google/adk/events/event.py` | `docs/design/events/event/index.md` |

If a document already exists at that path, update it in place and keep the
existing wording wherever the code has not changed, so the diff shows only what
the change actually altered.

`docs/design/` does not exist in this repository yet — the first design document
creates it. `docs/guides/` is the established sibling tree; match its directory
shape.

## Link to GitHub, not to local paths

These documents render on GitHub, where a local filesystem path is dead.
Rewrite `{repo_root}/src/google/adk/{topic}/{unit}.py#L93` as
`https://github.com/google/adk-python/blob/main/src/google/adk/{topic}/{unit}.py#L93`.

## Structure

Follow [references/design-template.md](references/design-template.md) section by
section.
