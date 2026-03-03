# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for ADK context management.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

from contextlib import aclosing
import inspect
from typing import Any
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import Union

# Re-export aclosing for backward compatibility
Aclosing = aclosing


def _is_context_type(annotation: Any) -> bool:
  """Check if an annotation is the Context type.

  This checks if the annotation is exactly Context or a type alias of Context
  (e.g., ToolContext, CallbackContext). Also handles Optional[Context] types.

  Args:
    annotation: The type annotation to check.

  Returns:
    True if the annotation is the Context type, False otherwise.
  """
  from ..agents.context import Context

  if annotation is inspect.Parameter.empty:
    return False

  # Handle Optional[Context] and Union types
  origin = get_origin(annotation)
  if origin is Union:
    args = get_args(annotation)
    return any(
        _is_context_type(arg) for arg in args if not isinstance(arg, type(None))
    )

  # Check if it's exactly the Context type (or an alias like ToolContext)
  return annotation is Context


def find_context_parameter(func: Callable[..., Any]) -> str | None:
  """Find the parameter name that has a Context type annotation.

  This function inspects the signature of a callable and returns the name
  of the first parameter that is annotated with Context or a type alias of
  Context (e.g., ToolContext, CallbackContext).

  Args:
    func: The callable to inspect.

  Returns:
    The parameter name if found, None otherwise.
  """
  if func is None:
    return None
  try:
    signature = inspect.signature(func)
  except (ValueError, TypeError):
    return None
  for name, param in signature.parameters.items():
    if _is_context_type(param.annotation):
      return name
  return None
