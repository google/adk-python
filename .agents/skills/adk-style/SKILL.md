---
name: adk-style
description: ADK development style guide for routine nits — Python idioms, codebase conventions, imports, typing, Pydantic patterns, formatting, logging, and file organization. Use this skill whenever writing code, tests, or reviewing PRs for the ADK project to ensure compliance with styling and coding conventions. Triggers on "code style", "how should I format", "naming convention", "lint", "nit", "imports", "typing", "Pydantic patterns", "testing rules".
---

# ADK Style Guide

## Style Guide (references/)
- [Visibility](references/visibility.md) — naming conventions for module-private, internal, and package-private visibility.
- [Imports](references/imports.md) — relative vs absolute imports, `TYPE_CHECKING` patterns.
- [Typing](references/typing.md) — strong typing, avoiding Any, bare type names, keyword-only arguments, `Optional` vs `| None`, abstract parameter types, mutable default avoidance, runtime type discrimination.
- [Pydantic Patterns](references/pydantic.md) — Pydantic v2 usage, `Field()` constraints, `field_validator`, `model_validator`, private attributes, deprecation migration, post-init setup.
- [Formatting](references/formatting.md) — indentation, line limits, and running pre-commit hooks.
- [Documentation](references/documentation.md) — comments and docstrings.
- [Logging](references/logging.md) — lazy evaluation and log levels.
- [File Organization](references/file-organization.md) — file headers and class organization.

## Testing
[references/testing.md](references/testing.md) — core principles, 9 rules for writing ADK tests, test structure template
