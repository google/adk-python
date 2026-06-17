# File Organization

- One class per file in `workflow/`.
- Private modules prefixed with `_` (e.g., `_base_node.py`).
- Public API exported through `__init__.py`.
- Unit tests must be placed in the same folder hierarchy under `tests/unittests/` as the original file in `src/`.
- If a single source file has multiple test files (e.g. testing different classes or behaviors separately), use the source file name (without leading underscores or extension) as the prefix for the test file names.
  - Example: `src/google/adk/tools/environment/_tools.py` -> `tests/unittests/tools/environment/test_tools_edit_file.py`

## File Headers

Every source file must have:
1. Apache 2.0 license header.
2. `from __future__ import annotations`.
3. Standard library imports, then third-party, then relative.
