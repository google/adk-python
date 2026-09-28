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

from __future__ import annotations

from typing import Any

"""Errors raised by the workflow framework."""


class NodeInterruptedError(BaseException):
  """Internal: raised when a dynamic node interrupts (HITL).

  Used exclusively by ``ctx.run_node()`` to signal that the dynamic
  child has unresolved interrupt IDs. The parent's NodeRunner catches
  this and reads the interrupt IDs from the parent's ctx (set by
  ``ctx.run_node()`` before raising).

  This is a ``BaseException`` so user code cannot accidentally catch
  it with ``except Exception``.

  Internal to the framework — not part of the public API.
  """


class GraphValidationError(ValueError):
  """Raised when a workflow graph is not well formed."""


class WorkflowConfigurationError(ValueError):
  """Raised when a workflow, node or edge is declared in an unusable way.

  The caller's construction mistake: the same declaration fails the same way
  every time, before anything runs.
  """


class WorkflowDataError(ValueError):
  """Raised when data arriving during a run does not fit what accepts it.

  Covers a node's inputs, an agent transfer target and an OAuth resume
  payload. The wiring is fine and only this run's data is wrong, so it can
  differ between runs of the same workflow.
  """


class WorkflowInvariantError(RuntimeError):
  """Raised when the framework reaches a state it is supposed to prevent.

  A bug in the engine rather than anything the caller did. Prefer this over
  ``assert``, which ``python -O`` removes.
  """


class NodeTimeoutError(Exception):
  """Raised when a node exceeds its configured timeout.

  This is a regular ``Exception`` (not ``BaseException``) so it is
  compatible with ``retry_config`` — a timed-out node can be retried.
  """

  def __init__(self, *, node_name: str, timeout: float) -> None:
    self.node_name = node_name
    self.timeout = timeout
    super().__init__(f"Node '{node_name}' timed out after {timeout} seconds.")


class DynamicNodeFailError(Exception):
  """Raised when a dynamic node fails.

  Caught by the parent node's NodeRunner to propagate the error.
  """

  def __init__(
      self, *, message: str, error: Exception, error_node_path: str
  ) -> None:
    self.error = error
    self.error_node_path = error_node_path
    # Surface the wrapped error's HTTP status/details so they aren't lost as the
    # error propagates up the execution stack. genai's client exposes `code`;
    # other libraries use `status_code`.
    self.status_code: Any | None = getattr(
        error, "status_code", getattr(error, "code", None)
    )
    details = getattr(error, "details", None)
    if details is None:
      # Raw httpx-style errors keep the body on the response instead.
      response = getattr(error, "response", None)
      if response is not None:
        details = getattr(response, "text", None)
    self.details: Any | None = details
    super().__init__(message)
