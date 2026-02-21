"""Graph state management with typed state and reducers."""

from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .state_utils import parse_state_value
from .state_utils import PydanticJSONEncoder
from .state_utils import state_value_as_dict
from .state_utils import state_value_as_str

# Re-export for backward compat
__all__ = ["GraphState", "StateReducer", "PydanticJSONEncoder"]

T = TypeVar("T", bound=BaseModel)


class StateReducer(str, Enum):
  """State reduction strategies for merging node outputs.

  Defines how node outputs are merged into the graph state:
  - OVERWRITE: Replace existing value with new value
  - APPEND: Append new value to list (creates list if needed)
  - SUM: Accumulate values using + operator (works for strings, numbers, lists)
  - CUSTOM: Use custom reducer function
  """

  OVERWRITE = "overwrite"
  APPEND = "append"
  SUM = "sum"
  CUSTOM = "custom"


class GraphState(BaseModel):  # type: ignore[misc]
  """Domain data container for graph execution.

  GraphState holds node outputs and intermediate results as the graph
  executes. Execution tracking (iteration, path, etc.) is handled
  separately by GraphAgentState.

  Example:
      ```python
      state = GraphState(
          data={"input": "user query", "result": "agent response"},
      )
      ```
  """

  model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

  data: Dict[str, Any] = Field(
      default_factory=dict, description="Node outputs and intermediate results"
  )

  def data_to_json(self, indent: int = 2) -> str:
    """Convert state.data to JSON string (automatically handles Pydantic models).

    Args:
        indent: JSON indentation level (default: 2)

    Returns:
        JSON string of state.data
    """
    import json

    return json.dumps(self.data, cls=PydanticJSONEncoder, indent=indent)

  def get_parsed(
      self, key: str, schema: Type[T], default: Optional[T] = None
  ) -> Optional[T]:
    """Get state value with automatic JSON-string parsing.

    Handles both dict and JSON-string representations transparently.

    Args:
        key: State data key (usually agent output_key)
        schema: Pydantic model to parse into
        default: Value to return if key missing or parse fails

    Returns:
        Parsed Pydantic model instance or default
    """
    return parse_state_value(self.data.get(key), schema, default)

  def get_str(self, key: str, default: str = "") -> str:
    """Get state value as string (for non-schema agent outputs).

    Args:
        key: State data key
        default: Value to return if key missing

    Returns:
        String value or default
    """
    return state_value_as_str(self.data.get(key), default)

  def get_dict(
      self, key: str, default: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Get state value as dict with JSON-string fallback.

    Args:
        key: State data key
        default: Value to return if key missing or parse fails

    Returns:
        Dict value or default
    """
    return state_value_as_dict(self.data.get(key), default)
