# Copyright 2025 Google LLC
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

from __future__ import annotations

import functools
import os
from typing import Callable
from typing import cast
from typing import Optional
from typing import TypeVar
from typing import Union
import warnings

from dotenv import load_dotenv

T = TypeVar("T", bound=Union[Callable, type])


def _make_feature_decorator(
    *,
    label: str,
    default_message: str,
    block_usage: bool = False,
    bypass_env_var: Optional[str] = None,
) -> Callable:
  def decorator_factory(message_or_obj=None):
    # Case 1: Used as @decorator without parentheses
    # message_or_obj is the decorated class/function
    if message_or_obj is not None and (
        isinstance(message_or_obj, type) or callable(message_or_obj)
    ):
      return _create_decorator(
          default_message, label, block_usage, bypass_env_var
      )(message_or_obj)

    # Case 2: Used as @decorator() with or without message
    # message_or_obj is either None or a string message
    message = (
        message_or_obj if isinstance(message_or_obj, str) else default_message
    )
    return _create_decorator(message, label, block_usage, bypass_env_var)

  return decorator_factory


def _create_decorator(
    message: str, label: str, block_usage: bool, bypass_env_var: Optional[str]
) -> Callable[[T], T]:
  def decorator(obj: T) -> T:
    obj_name = getattr(obj, "__name__", type(obj).__name__)
    msg = f"[{label.upper()}] {obj_name}: {message}"

    if isinstance(obj, type):  # decorating a class
      orig_init = obj.__init__

      @functools.wraps(orig_init)
      def new_init(self, *args, **kwargs):
        # Load .env file if dotenv is available
        load_dotenv()

        # Check if usage should be bypassed via environment variable at call time
        should_bypass = (
            bypass_env_var is not None
            and os.environ.get(bypass_env_var, "").lower() == "true"
        )

        if should_bypass:
          # Bypass completely - no warning, no error
          pass
        elif block_usage:
          raise RuntimeError(msg)
        else:
          warnings.warn(msg, category=UserWarning, stacklevel=2)
        return orig_init(self, *args, **kwargs)

      obj.__init__ = new_init  # type: ignore[attr-defined]
      return cast(T, obj)

    elif callable(obj):  # decorating a function or method

      @functools.wraps(obj)
      def wrapper(*args, **kwargs):
        # Load .env file if dotenv is available
        load_dotenv()

        # Check if usage should be bypassed via environment variable at call time
        should_bypass = (
            bypass_env_var is not None
            and os.environ.get(bypass_env_var, "").lower() == "true"
        )

        if should_bypass:
          # Bypass completely - no warning, no error
          pass
        elif block_usage:
          raise RuntimeError(msg)
        else:
          warnings.warn(msg, category=UserWarning, stacklevel=2)
        return obj(*args, **kwargs)

      return cast(T, wrapper)

    else:
      raise TypeError(
          f"@{label} can only be applied to classes or callable objects"
      )

  return decorator


working_in_progress = _make_feature_decorator(
    label="WIP",
    default_message=(
        "This feature is a work in progress and is not working completely. ADK"
        " users are not supposed to use it."
    ),
    block_usage=True,
    bypass_env_var="ADK_ALLOW_WIP_FEATURES",
)
"""Mark a class or function as a work in progress.

By default, decorated functions/classes will raise RuntimeError when used.
Set ADK_ALLOW_WIP_FEATURES=true environment variable to bypass this restriction.
ADK users are not supposed to set this environment variable.

Sample usage:

```
@working_in_progress("This feature is not ready for production use.")
def my_wip_function():
  pass
```
"""

experimental = _make_feature_decorator(
    label="EXPERIMENTAL",
    default_message=(
        "This feature is experimental and may change or be removed in future"
        " versions without notice. It may introduce breaking changes at any"
        " time."
    ),
)
"""Mark a class or a function as an experimental feature.

Sample usage:

```
# Use with default message
@experimental
class ExperimentalClass:
  pass

# Use with custom message
@experimental("This API may have breaking change in the future.")
class CustomExperimentalClass:
  pass

# Use with empty parentheses (same as default message)
@experimental()
def experimental_function():
  pass
```
"""
