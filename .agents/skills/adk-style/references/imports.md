# Imports Style Guide

## General Rules

- **Source code** (`src/`): use relative imports.
  `from ..agents.llm_agent import LlmAgent`
- **Tests** (`tests/`): use absolute imports.
  `from google.adk.agents.llm_agent import LlmAgent`
- **Import from the module, not the package.** `from ..agents.llm_agent import
  LlmAgent`, never `from ..agents import LlmAgent`. Importing through
  `__init__.py` inside the framework creates import cycles and forces the
  package to eagerly load unrelated modules.
- **CLI package** (`cli/`):
  - Treat it as an external package.
  - Use **relative imports** for files within `cli/`.
  - Use **absolute imports** for files outside `cli/`.
  - **Dependency direction**: `cli/` may import from the rest of the codebase;
    nothing outside `cli/` may import from it. `scripts/compliance_checks.py`
    fails the commit on any `from ...cli... import ...` outside the package.

## One name per line

`isort`'s `google` profile puts every imported name on its own line, so
`from typing import Any, Optional` becomes two lines. Ordering is
case-insensitive and does not group by type, which is why `PrivateAttr` sorts
after `model_validator`:

```python
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import PrivateAttr
```

Three groups, blank-line separated: standard library, third party, then
relative. In tests, `google.adk` sorts into the third-party group
(`known_third_party` in `pyproject.toml`), alongside `google.genai` and
`pytest`.

## Don't wrap a long import

The 80-character limit does not apply to imports: `isort` is configured with
`line_length = 200` and pyink leaves import lines intact. A long
`from ... import ...` stays on one line — the codebase has no parenthesized
from-imports. Adding parentheses or a line break will be reverted by the next
format run.

## TYPE_CHECKING Imports

Use `TYPE_CHECKING` for imports needed only by type hints, to avoid circular
imports at runtime:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
```

This works because `from __future__ import annotations` makes annotations
strings (deferred evaluation), so the import is never needed at runtime.
