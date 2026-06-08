# Visibility Style Guide

Python does not have native access modifiers (like `public`, `private`, or `package-private`). ADK relies on naming conventions and module structure to define visibility boundaries.

## Conventions

### 1. Module-Private / Internal Files

- **Private by Default**: All new `.py` module files under `src/google/adk/` must be private by default (prefixed with `_`). This is enforced by a pre-commit hook (`check-new-py-prefix`).
- Even if a file contains symbols intended for the public API, the file itself must have a leading underscore. The symbols are then exposed via the package's `__init__.py`.
- Files intended for internal use within a package or subsystem must also be prefixed with a leading underscore (e.g., `_task_models.py`).
- These files should **never** be imported directly by code outside of the ADK framework.

### 2. Class and Function Visibility

- **Public**: No leading underscore. Intended for use by consumers of the module or package.
- **Internal/Private**: Leading underscore (e.g., `_private_method()`). Intended only for use within the defining class or module.

### 3. Package-Private (Subsystem Visibility)

Since Python lacks true package-private access, we simulate it by:

- **Not exporting** the symbol in the package's `__init__.py`.
- Using `_`-prefixed modules for internal implementation details.
- Code within the same package can import from these `_` modules, but code outside should not.
- **Direct Imports Required**: Within the ADK framework, importing from `__init__.py` is **not allowed**. You must import from the specific module directly. This helps keep `__init__.py` minimal and keeps packages as self-contained as possible.

### 4. Public API Export

- The public API of a package must be explicitly exported in `__init__.py`.
- **Use `__all__`**: The `__init__.py` file should define `__all__` to explicitly list the symbols that are part of the public API.
- **Only public names** (symbols intended for use outside the package) should be imported into `__init__.py` and listed in `__all__`.
- Users should be able to import public symbols directly from the package level, rather than digging into internal modules.

## Examples

### Exposing a Public Interface

```python
# In src/google/adk/agents/llm/task/_task_agent.py (File is private by default)
class TaskAgent: # Public symbol
    ...

# In src/google/adk/agents/llm/task/__init__.py
from ._task_agent import TaskAgent

__all__ = [
    'TaskAgent',
]
```

### Keeping Implementation Details Private

```python
# In src/google/adk/agents/llm/task/_task_models.py (Internal file)
class TaskRequest(BaseModel): # Public within the module, but module is private
    ...

# In src/google/adk/agents/llm/task/__init__.py
# We DO NOT export TaskRequest here if it is only for internal use within the task package.
```
