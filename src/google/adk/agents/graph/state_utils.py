"""Reusable state parsing utilities.

Generic functions for parsing state values — usable by any agent type,
not just GraphAgent. GraphState.get_parsed/get_str/get_dict delegate here.

Example:
    ```python
    from google.adk.agents.graph.state_utils import parse_state_value

    # Parse a raw value (dict or JSON string) into a Pydantic model
    result = parse_state_value(raw_value, MyModel)
    ```
"""

from __future__ import annotations

import json
from typing import Any
from typing import cast
from typing import Dict
from typing import Optional
from typing import Type
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticJSONEncoder(json.JSONEncoder):
  """JSON encoder that automatically handles Pydantic models.

  This encoder allows json.dumps() to work transparently with Pydantic models
  without requiring special serialization methods.

  Example:
      ```python
      import json
      state_json = json.dumps(data, cls=PydanticJSONEncoder, indent=2)
      ```
  """

  def default(self, obj: Any) -> Any:
    """Convert Pydantic models to dicts automatically."""
    if isinstance(obj, BaseModel):
      return obj.model_dump()
    return super().default(obj)


def parse_state_value(
    raw: Any, schema: Type[T], default: Optional[T] = None
) -> Optional[T]:
  """Parse a raw state value into a Pydantic model.

  Handles both dict and JSON-string representations transparently.

  Args:
      raw: Raw value from state (dict, JSON string, or None)
      schema: Pydantic model class to parse into
      default: Value to return if raw is None or parse fails

  Returns:
      Parsed Pydantic model instance or default
  """
  if raw is None:
    return default

  if isinstance(raw, dict):
    try:
      return cast(T, schema.model_validate(raw))
    except Exception:
      return default

  if isinstance(raw, str):
    try:
      return cast(T, schema.model_validate_json(raw))
    except Exception:
      return default

  return default


def state_value_as_str(raw: Any, default: str = "") -> str:
  """Convert a raw state value to string.

  Args:
      raw: Raw value from state
      default: Value to return if raw is None

  Returns:
      String representation or default
  """
  if raw is None:
    return default
  return str(raw)


def state_value_as_dict(
    raw: Any, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
  """Convert a raw state value to dict, with JSON-string fallback.

  Args:
      raw: Raw value from state (dict, JSON string, or other)
      default: Value to return if conversion fails

  Returns:
      Dict value or default
  """
  _default = default or {}

  if raw is None:
    return _default

  if isinstance(raw, dict):
    return cast(Dict[str, Any], raw)

  if isinstance(raw, str):
    try:
      result = json.loads(raw)
      if isinstance(result, dict):
        return cast(Dict[str, Any], result)
      return _default
    except Exception:
      return _default

  return _default
