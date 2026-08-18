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

"""Platform module for abstracting random number generation."""

from __future__ import annotations

from contextvars import ContextVar
import random
from typing import Callable

_default_random: random.Random = random.Random()
_default_random_provider: Callable[[], random.Random] = lambda: _default_random
_random_provider_context_var: ContextVar[Callable[[], random.Random]] = (
    ContextVar("random_provider", default=_default_random_provider)
)


def set_random_provider(provider: Callable[[], random.Random]) -> None:
  """Sets the provider for the random number generator.

  Args:
    provider: A callable that returns the `random.Random` instance to use. Note
      that the provider callable is evaluated on every `get_random()` call; to
      preserve RNG sequence state across calls, return an existing
      `random.Random` instance rather than constructing a new one inside the
      callable (e.g., `rng = random.Random(42); set_random_provider(lambda:
      rng)` instead of `set_random_provider(lambda: random.Random(42))`).
  """
  _random_provider_context_var.set(provider)


def reset_random_provider() -> None:
  """Resets the random provider to its default implementation."""
  _random_provider_context_var.set(_default_random_provider)


def get_random() -> random.Random:
  """Returns the random number generator."""
  return _random_provider_context_var.get()()
