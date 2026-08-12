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

"""Asserting that every async generator a scenario iterates is closed.

Necessary because instrumentation utilizes contextvars, which run into
"ContextVar was created in a different Context" errors when a given
coroutine gets indeterminately suspended.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from contextlib import aclosing
from contextlib import contextmanager
import gc
import inspect
import sys
from types import CodeType

# ---------------------------------------------------------------------------
# aclosing wrapping assertions.
# ---------------------------------------------------------------------------


@contextmanager
def aclosing_wrapping_assertions() -> Iterator[None]:
  """Context manager that asserts every async generator is wrapped in ``aclosing``.

  The check uses ``gc.get_referrers`` on every async generator first
  iterated within the block, which is expensive (~5 seconds per
  scenario). Run this once per scenario rather than per parametrized
  test case.

  On exit the original ``sys`` async-gen hooks are restored.
  """
  prev_firstiter, prev_finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro: AsyncGenerator[object, object]):
    if _is_async_context_manager():
      if prev_firstiter:
        prev_firstiter(coro)
      return

    assert any(
        isinstance(referrer, aclosing)
        or isinstance(indirect_referrer, aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in Python 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), _no_aclosing_assertion_error(coro)

    if prev_firstiter:
      prev_firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, prev_finalizer)
  try:
    yield
  finally:
    sys.set_asyncgen_hooks(prev_firstiter, prev_finalizer)


def _no_aclosing_assertion_error(coro: AsyncGenerator[object, object]) -> str:
  first_iter_loc = ""
  definition_loc = ""

  if (f := inspect.currentframe()) and (f := f.f_back) and (f := f.f_back):
    first_iter_loc = f'file "{f.f_code.co_filename}" line "{f.f_lineno}"'
  if (ag_code := getattr(coro, "ag_code", None)) and isinstance(
      ag_code, CodeType
  ):
    definition_loc = (
        f'file "{ag_code.co_filename}" line "{ag_code.co_firstlineno}"'
    )

  header_str = f'Async generator "{coro.__name__}" is not wrapped in aclosing'
  first_iter_str = (
      f"first iterated in {first_iter_loc}" if first_iter_loc else ""
  )
  definition_str = f"defined in {definition_loc}" if definition_loc else ""
  instruction_str = """
Wrap the iteration in the following code snippet before iterating:

async with contextlib.aclosing(...) as agen:
  async for ... as agen:
     ...
"""

  return "\n".join(
      part
      for part in [
          header_str,
          first_iter_str,
          definition_str,
          instruction_str,
      ]
      if part
  )


def _is_async_context_manager() -> bool:
  """Checks if this function was invoked by contextlib.asynccontextmanager."""
  frame = inspect.currentframe()
  while frame:
    if (
        frame.f_code.co_name == "__aenter__"
        and "contextlib" in frame.f_code.co_filename
    ):
      return True
    frame = frame.f_back
  return False
