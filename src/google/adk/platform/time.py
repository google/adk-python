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

"""Platform module for abstracting system time generation."""

from __future__ import annotations

from contextvars import ContextVar
import time
from typing import Callable

_default_time_provider: Callable[[], float] = time.time
_time_provider_context_var: ContextVar[Callable[[], float]] = ContextVar(
    "time_provider", default=_default_time_provider
)


def set_time_provider(provider: Callable[[], float]) -> None:
  """Sets the provider for the current time.

  Args:
    provider: A callable that returns the current time in seconds since the
      epoch.
  """
  _time_provider_context_var.set(provider)


def reset_time_provider() -> None:
  """Resets the time provider to its default implementation."""
  _time_provider_context_var.set(_default_time_provider)


def get_time() -> float:
  """Returns the current time in seconds since the epoch."""
  return _time_provider_context_var.get()()
