---
name: adk-style
description: >-
  Python style and codebase conventions for ADK (Agent Development Kit):
  private-by-default file visibility, imports, type hints, Pydantic v2 models,
  formatting, docstrings, logging, async I/O, file and test layout, and unit
  test structure. Use when writing or editing ADK source or tests, deciding
  whether a new file or symbol should be public or private, naming or placing
  a test file, fixing a formatter, linter, or type-check failure (pyink,
  isort, ruff, mypy, addlicense, compliance-checks), or asking whether code
  matches house style. Don't use for reviewing a whole changeset (use
  adk-review), writing a developer guide or design doc for a code unit (use
  adk-unit-guide or adk-unit-design), building or configuring agents (use
  adk-agent-builder), or installing the toolchain (use adk-setup).
---

# ADK Style Guide

Conventions for `src/google/adk/` and `tests/unittests/`. Most are enforced by
a pre-commit hook or a CI job, so a violation blocks the PR rather than
surfacing in review. Read the one reference for the topic you are touching.

## Pick a reference

| Task | Reference |
| --- | --- |
| Adding a `.py` file; deciding public vs private; `__init__.py` and `__all__` | [visibility.md](references/visibility.md) |
| Writing `import` lines; relative vs absolute; circular imports; `TYPE_CHECKING` | [imports.md](references/imports.md) |
| Annotating args and returns; `Optional` vs `\| None`; keyword-only args; `isinstance`; asserts; mypy | [typing.md](references/typing.md) |
| Defining a Pydantic model, validator, private attribute, or on-wire payload | [pydantic.md](references/pydantic.md) |
| Indentation, line length, quotes; running the formatter; what each hook checks | [formatting.md](references/formatting.md) |
| Writing a docstring or an explanatory comment | [documentation.md](references/documentation.md) |
| Emitting a log record; naming the module logger; picking a level | [logging.md](references/logging.md) |
| Anything that performs I/O — network, disk, database | [async.md](references/async.md) |
| Where a new file goes; license header; where its test goes and what to call it | [file-organization.md](references/file-organization.md) |
| Writing or restructuring a unit test | [testing.md](references/testing.md) |

## A check failed — where to look

| Failing check | Reference |
| --- | --- |
| `check-new-py-prefix` | [visibility.md](references/visibility.md) |
| `compliance-checks` | [logging.md](references/logging.md) (logger name), [typing.md](references/typing.md) (`from __future__ import annotations`), [imports.md](references/imports.md) (`cli/` import direction) |
| `pyink`, `isort`, `ruff`, `addlicense`, `codespell` | [formatting.md](references/formatting.md) |
| Mypy Check CI job | [typing.md](references/typing.md) |
