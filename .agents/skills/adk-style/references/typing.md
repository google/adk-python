# Type Hints and Strong Typing

## General Rules

- **Prefer Strong Typing**: Use type hints for all function arguments and return types. Avoid leaving types unspecified.
- **Minimize `Any`**: Use specific types or `Generic` whenever possible. Avoid `Any` as it bypasses type checking.
- **No double-quoted type hints**: When `from __future__ import annotations` is present, use bare type names (e.g., `list[str]` instead of `"list[str]"`).
- **Always include `from __future__ import annotations`**: Every source file must include this immediately after the license header, before any other imports. This enables forward-referencing classes without quotes (PEP 563).

## `Optional[X]` vs `X | None`

The codebase uses both styles. Follow this convention:

- **New code** (especially in `workflow/`): Prefer `X | None` — it is more concise and modern.
- **Existing files**: Match the style already used in the file for consistency.
- **Both are acceptable** — do not refactor one to the other without reason.


## Abstract Types for Function Parameters

Use abstract types from `collections.abc` for function parameter annotations. This accepts the widest range of inputs while remaining type-safe. Use concrete types for return annotations to give callers the most useful information.



## Keyword-Only Arguments

Use `*` to force keyword-only arguments on functions with multiple parameters of the same type, or where argument order is error-prone. This is a widely used pattern in ADK (16+ files).


**When to use `*`:**
- Constructors (`__init__`) with 2+ non-self parameters
- Any function where swapping arguments would silently produce wrong results
- Methods with multiple `str` or `int` parameters

## Mutable Default Arguments

**Never use mutable default arguments.** Use `None` as a sentinel and initialize in the function body. This is a well-followed pattern throughout ADK.


This applies to `list`, `dict`, `set`, and any other mutable type.

## Runtime Type Discrimination with `isinstance()`

Use `isinstance()` for runtime type discrimination when handling polymorphic inputs. This is pervasive in ADK (700+ usages). Prefer exhaustive `if/elif` chains with a clear fallback.


**Guidelines:**
- Always include an `else` branch that raises `TypeError` or handles the unknown case.
- Prefer `isinstance(x, SomeType)` over `type(x) is SomeType` — it handles subclasses correctly.
- For checking multiple types: `isinstance(x, (TypeA, TypeB))`.
