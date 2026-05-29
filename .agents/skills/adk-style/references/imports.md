# Imports Style Guide

## General Rules

- **Source code** (`src/`): Use relative imports.
  `from ..agents.llm_agent import LlmAgent`
- **Tests** (`tests/`): Use absolute imports.
  `from google.adk.agents.llm_agent import LlmAgent`
- **Import from module**: Import from the module file, not from `__init__.py`.
  `from ..agents.llm_agent import LlmAgent` (not `from ..agents import LlmAgent`)
- **CLI package** (`cli/`):
  - Treat as an external package.
  - Use **relative imports** for files within the `cli/` package.
  - Use **absolute imports** for files outside of the `cli/` package.
  - **Dependency Direction**: Only `cli/` can import from the rest of the codebase. The other codebase must **STRICTLY NOT** import from `cli/`.

## TYPE_CHECKING Imports

Use `TYPE_CHECKING` for imports needed only by type hints to avoid circular imports at runtime:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.invocation_context import InvocationContext
```

This works because `from __future__ import annotations` makes all annotations strings (deferred evaluation), so the import is never needed at runtime.
