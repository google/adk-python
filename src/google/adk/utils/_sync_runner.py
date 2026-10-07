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

"""Binds the executor that runs synchronous user callables.

The tool caller binds a runner for the duration of a tool call when the tool
thread pool is enabled. FunctionTool and FunctionNode read the binding and hand
their synchronous callables to the runner instead of calling them inline. The
binding is a context variable, so it follows the call into nested tasks and
nodes without being threaded through as a parameter.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Iterator

_SyncCallableRunner = Callable[
    [Callable[..., Any], dict[str, Any]], Awaitable[Any]
]
_SYNC_CALLABLE_RUNNER: contextvars.ContextVar[_SyncCallableRunner | None] = (
    contextvars.ContextVar("adk_sync_callable_runner", default=None)
)


@contextmanager
def _use_sync_callable_runner(
    runner: _SyncCallableRunner | None = None,
) -> Iterator[None]:
  """Binds the runner used for synchronous callables.

  Passing ``None`` clears the binding, which stops a worker-owned nested call
  from reusing the caller's runner.
  """
  token = _SYNC_CALLABLE_RUNNER.set(runner)
  try:
    yield
  finally:
    _SYNC_CALLABLE_RUNNER.reset(token)
