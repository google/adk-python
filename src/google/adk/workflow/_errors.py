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
    super().__init__(message)
