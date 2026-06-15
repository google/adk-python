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

"""Platform module for abstracting unique ID generation."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable
import uuid

_default_id_provider: Callable[[], str] = lambda: str(uuid.uuid4())
_id_provider_context_var: ContextVar[Callable[[], str]] = ContextVar(
    "id_provider", default=_default_id_provider
)


def set_id_provider(provider: Callable[[], str]) -> None:
  """Sets the provider for generating unique IDs.

  Args:
    provider: A callable that returns a unique ID string.
  """
  _id_provider_context_var.set(provider)


def reset_id_provider() -> None:
  """Resets the ID provider to its default implementation."""
  _id_provider_context_var.set(_default_id_provider)


def new_uuid() -> str:
  """Returns a new unique ID."""
  return _id_provider_context_var.get()()
