# Pydantic Patterns

ADK models use Pydantic v2. This guide covers the key patterns used throughout the codebase.

## Basic Model Structure

- Use `Field()` for validation, defaults, and descriptions.
- Use `PrivateAttr()` for internal state that shouldn't be serialized.
- Use `model_post_init()` instead of `__init__` for setup logic.
- Prefer `model_dump()` over `dict()` (Pydantic v2).

## On-Wire Models

For Pydantic models that cross network or system boundaries (e.g., API payloads, WebSocket messages, event persistence), inherit from `SerializedBaseModel` located in `google.adk.utils._serialized_base_model`.

This ensures:
- camelCase serialization by default (via `alias_generator=to_camel`).

## Docstrings as Field Descriptions

To keep code Pythonic and ensure that generated schemas stay in sync with documentation, it is **strongly recommended** to use docstrings as field descriptions for all Pydantic models in the ADK codebase.

To enable this, add `use_attribute_docstrings=True` to your model's `ConfigDict`:

```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    field_name: str
    """Description of the field."""
```

Note: If you are inheriting from `SerializedBaseModel`, this is already enabled by default.

## Summary of When to Use Each

| Need | Pattern |
|---|---|
| Simple numeric/string bounds | `Field(ge=0, le=100)` |
| Single-field business logic | `@field_validator('field', mode='after')` |
| Cross-field consistency | `@model_validator(mode='after')` |
| Field deprecation/migration | `@model_validator(mode='before')` |
| Internal mutable state | `PrivateAttr(default_factory=...)` |
| Post-construction setup | `model_post_init()` |

## `Field()` with Constraints

Use `Field()` constraints for declarative validation directly on the field definition. This keeps validation close to the data declaration and avoids custom validator boilerplate.



## `field_validator` — Single-Field Validation

Use `@field_validator` for validation logic that goes beyond simple constraints. This is heavily used in ADK (36+ instances). Always use `mode='after'` unless you need to intercept raw input before Pydantic coercion.


**Rules:**
- Decorate with `@field_validator(...)`. While `@classmethod` is automatically applied by Pydantic v2, adding it is recommended in ADK for explicit visibility.
- Return the (possibly transformed) value.
- Raise `ValueError` with a descriptive message on failure.
- Prefer `mode='after'` (validates after Pydantic's own parsing/coercion).

## `model_validator` — Cross-Field and Migration Validation

Use `@model_validator` when validation depends on multiple fields, or when handling deprecation/migration of field names.

### `mode='before'` — Deprecation and Field Migration


### `mode='after'` — Cross-Field Consistency


**Rules:**
- `mode='before'`: receives raw `data` (usually `dict`). Use for field renaming, deprecation, and input normalization. Must return the (modified) data.
- `mode='after'`: receives the fully constructed model instance (`self`). Use for cross-field consistency checks. Must return `self`.
- Always guard `mode='before'` validators with `isinstance(data, dict)` since data could also come as an existing model instance.
