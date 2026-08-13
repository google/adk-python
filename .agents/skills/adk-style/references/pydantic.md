# Pydantic Patterns

ADK models use Pydantic v2.

## Basic Model Structure

- Use `Field()` for validation, defaults, and descriptions.
- Use `PrivateAttr()` for internal state that must not be serialized.
- Use `model_post_init()` for setup logic, not `__init__` — overriding
  `__init__` on a Pydantic model bypasses validation ordering.
- Use `model_dump()` / `model_dump_json()`, not the v1 `dict()` / `json()`.

## Which mechanism to use

| Need | Pattern |
| --- | --- |
| Simple numeric/string bounds | `Field(ge=0, le=100)` |
| Single-field business logic | `@field_validator('field')` |
| Cross-field consistency | `@model_validator(mode='after')` |
| Field deprecation/migration | `@model_validator(mode='before')` |
| Internal mutable state | `PrivateAttr(default_factory=...)` |
| Post-construction setup | `model_post_init()` |

## `Field()` with Constraints

Declare bounds on the field rather than writing a validator for them — it
keeps the rule next to the data and shows up in the generated JSON schema.

```python
compaction_interval: Optional[int] = Field(default=None, gt=0)
injected_latency_seconds: float = Field(default=0.0, le=120.0)
```

## Documenting fields

The house style is an attribute docstring directly under the field, which
Sphinx picks up:

```python
model: Union[str, BaseLlm] = ''
"""The model to use for the agent.

When not set, the agent inherits the model from its ancestor.
"""
```

Those docstrings are documentation only. To also make them the field
descriptions in the generated JSON schema, set `use_attribute_docstrings=True`
in the model's `ConfigDict`; `SerializedBaseModel` already enables it.

```python
class MyModel(BaseModel):
  model_config = ConfigDict(use_attribute_docstrings=True)

  field_name: str
  """Description of the field."""
```

## On-Wire Models

A model that crosses a network or storage boundary — an API payload, a
WebSocket message, a persisted event — should inherit from
`SerializedBaseModel` in `google.adk.utils._serialized_base_model` rather than
`BaseModel`. It sets `alias_generator=to_camel` with `populate_by_name=True`
and defaults `model_dump_json()` to `by_alias=True`, so Python stays
snake_case while the wire format stays camelCase without every call site
remembering to pass `by_alias`.

## `field_validator` — Single-Field Validation

Use `@field_validator` when a constraint needs logic that `Field()` cannot
express.

```python
@field_validator('max_llm_calls')
@classmethod
def validate_max_llm_calls(cls, value: int) -> int:
  if value <= 0:
    raise ValueError('max_llm_calls must be positive.')
  return value
```

**Rules:**

- Add `@classmethod` under the decorator. Pydantic v2 applies it implicitly,
  but ADK writes it out — every validator in the codebase does.
- Return the (possibly transformed) value.
- Raise `ValueError` with a message that names the field and the bound.
- The default mode is `'after'`, which runs post-coercion and is what you
  almost always want; omit the argument. Pass `mode='before'` only to
  intercept raw input.

## `model_validator` — Cross-Field and Migration Validation

### `mode='before'` — deprecation and field migration

Receives the raw input, usually a `dict`, before any field is parsed. Use it
to rename or back-fill fields.

```python
@model_validator(mode='before')
@classmethod
def check_for_deprecated_save_live_audio(cls, data: Any) -> Any:
  """If save_live_audio is passed, use it to set save_live_blob."""
  if isinstance(data, dict) and 'save_live_audio' in data:
    warnings.warn(
        'The `save_live_audio` config is deprecated, use `save_live_blob`.',
        DeprecationWarning,
        stacklevel=2,
    )
    if data['save_live_audio']:
      data['save_live_blob'] = True
  return data
```

Guard with `isinstance(data, dict)`: the input can also arrive as an already
constructed model instance, and indexing that raises.

### `mode='after'` — cross-field consistency

Receives the constructed instance and must return it.

```python
@model_validator(mode='after')
def _validate_parallel_worker_config(self) -> Node:
  if self.max_parallel_workers is not None and not self.parallel_worker:
    raise ValueError(
        'max_parallel_workers can only be set when parallel_worker is True.'
    )
  return self
```
