# File Organization

- One class per file in `workflow/`.
- New modules under `src/google/adk/` are private by default — see
  the visibility reference for the naming rules and how to expose a
  public symbol.

## File Headers

Every module under `src/google/adk/` starts with:

1. The Apache 2.0 license header (added by the `addlicense` hook).
2. `from __future__ import annotations`.
3. Imports: standard library, third party, then relative.

`from __future__ import annotations` is checked by
`scripts/compliance_checks.py`, which exempts `__init__.py`, `version.py`,
`tests/`, and `contributing/samples/`.

## Where tests go

Mirror the source path under `tests/unittests/`:

```text
src/google/adk/tools/environment/_edit_file_tool.py
tests/unittests/tools/environment/test_edit_file_tool.py
```

When one source file needs several test files, use the source file name —
without the leading underscore or extension — as a shared prefix:

```text
src/google/adk/workflow/_workflow.py
tests/unittests/workflow/test_workflow.py
tests/unittests/workflow/test_workflow_hitl.py
tests/unittests/workflow/test_workflow_nested.py
```
