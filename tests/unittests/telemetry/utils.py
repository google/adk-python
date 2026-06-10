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

from collections.abc import AsyncGenerator
from contextlib import aclosing
import gc
import inspect
import sys
from types import CodeType
from typing import Any


def set_aclosing_wrapping_assertions():
  firstiter, finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro: AsyncGenerator[Any, Any]):
    nonlocal firstiter

    if _is_async_context_manager():
      if firstiter:
        firstiter(coro)
      return

    assert any(
        isinstance(referrer, aclosing)
        or isinstance(indirect_referrer, aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in Python 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), _no_aclosing_assertion_error(coro)

    if firstiter:
      firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, finalizer)


def _no_aclosing_assertion_error(coro: AsyncGenerator[Any, Any]):
  first_iter_loc = ""
  definition_loc = ""

  # Get frame where the async generator was first called.
  # 1. currentframe returns `_no_aclosing_assertion_error` (current function) frame.
  # 2. First f.back returns `wrapped_firstiter` frame.
  # 3. Second f.back returns code location where the generator is first iterated,
  # where wrapping in aclosing should happen.
  if (f := inspect.currentframe()) and (f := f.f_back) and (f := f.f_back):
    first_iter_loc = f'file "{f.f_code.co_filename}" line "{f.f_lineno}"'
  # In case the code location of first iteration is missing or incorrect,
  # the place where async generator is defined is useful, because
  # it's possible to iterate through references of the async generator.
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


def _is_async_context_manager():
  """Checks if this function was invoked by contextlib.asynccontextmanager.

  contextlib.asynccontextmanager is implemented on top of async generators.
  We don't need to however check if these are wrapped in aclosing, because
  they cannot be interrupted midway through their execution if
  all async generators in the application flow are wrapped in aclosing.
  """
  frame = inspect.currentframe()
  while frame:
    if (
        frame.f_code.co_name == "__aenter__"
        and "contextlib" in frame.f_code.co_filename
    ):
      return True
    frame = frame.f_back
  return False
