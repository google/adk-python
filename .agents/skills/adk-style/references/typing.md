# Type Hints and Strong Typing

## General Rules

- **Annotate everything**: type hints on all function arguments and return
  types.
- **Minimize `Any`**: use a specific type or a `TypeVar`. `Any` disables
  checking for every value that flows through it.
- **`from __future__ import annotations` goes at the top of every module**
  under `src/google/adk/`, immediately after the license header and before any
  other import. `scripts/compliance_checks.py` fails the commit if it is
  missing. Exempt: `__init__.py`, `version.py`, `tests/`, and
  `contributing/samples/`.
- **No quoted type hints.** Deferred annotations make forward references work
  unquoted, so write `list[str]`, not `"list[str]"`.
- **Builtin generics** for new code: `list[str]`, `dict[str, int]`,
  `tuple[str, ...]`. `typing.List` / `typing.Dict` survive in older modules;
  don't add more, and don't churn existing ones.

## Mypy

Mypy runs in `strict` mode against `src/` with the Pydantic plugin, targeting
Python 3.11 (`[tool.mypy]` in `pyproject.toml`). `tests/` and
`contributing/samples/` are excluded.

```bash
mypy .
```

The CI job compares your branch's errors against the base branch and fails
only on **new** ones, so a pre-existing error in a file you touched is not
your problem — an error on a line you added is.

## `Optional[X]` vs `X | None`

Both appear in the codebase. Follow this convention:

- **New code** (especially in `workflow/`): prefer `X | None`.
- **Existing files**: match the style already in the file.
- Do not refactor one into the other without a reason.

## Abstract Types for Function Parameters

Annotate parameters with abstract types from `collections.abc` so callers can
pass any compatible container; annotate returns with the concrete type so
callers know exactly what they get.

```python
from collections.abc import Mapping
from collections.abc import Sequence

def merge_labels(
    labels: Mapping[str, str], extra: Sequence[str]
) -> dict[str, str]:
  ...
```

## Keyword-Only Arguments

Put `*` before the parameters of any constructor or function where argument
order is easy to get wrong — two parameters of the same type is enough for a
silent bug.

```python
class NodeRunner:

  def __init__(
      self,
      *,
      node: BaseNode,
      parent_ctx: Context,
      run_id: str | None = None,
  ):
    ...
```

Use it for: constructors with 2+ non-`self` parameters, any function where
swapping two arguments would still typecheck, and methods taking several
`str` or `int` parameters.

## Mutable Default Arguments

A mutable default is evaluated once at definition time and shared by every
call, so one caller's mutation leaks into the next. Use `None` as a sentinel:

```python
# Bad — every caller shares one list.
def add(item: str, items: list[str] = []) -> list[str]:
  ...

# Good
def add(item: str, items: list[str] | None = None) -> list[str]:
  items = list(items) if items else []
  ...
```

This applies to `list`, `dict`, `set`, and any other mutable type.

## Runtime Type Discrimination with `isinstance()`

`isinstance()` is the codebase's standard way to handle polymorphic input.
Write exhaustive `if`/`elif` chains and always terminate them:

```python
if isinstance(node, FunctionNode):
  ...
elif isinstance(node, (JoinNode, ToolNode)):
  ...
else:
  raise TypeError(f'Unsupported node type: {type(node)}')
```

- Always include an `else` that raises `TypeError` or handles the unknown
  case, so a new subclass fails loudly instead of silently doing nothing.
- Prefer `isinstance(x, SomeType)` over `type(x) is SomeType` — it handles
  subclasses.
- Check several types at once with a tuple: `isinstance(x, (TypeA, TypeB))`.

## No Asserts in Production Code

`assert` is stripped when Python runs with `-O`, so an assertion is not a
runtime guarantee, and its failure message tells the caller nothing. Raise
`ValueError`, `TypeError`, or `RuntimeError` instead. Asserts in tests are
fine.
