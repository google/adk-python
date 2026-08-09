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

"""Tests for the workflow error types."""

from google.adk.workflow._errors import NodeInterruptedError
import pytest


def test_node_interrupted_error_survives_a_broad_except_in_node_code():
  """A node pausing for human input must not be swallowed by user code.

  Node bodies routinely wrap their work in ``except Exception``. If an
  interrupt were catchable there, the pause would be converted into a normal
  return and the node would be recorded as completed instead of waiting.
  """

  def node_body_that_swallows_errors():
    try:
      raise NodeInterruptedError()
    except Exception:  # pylint: disable=broad-except
      return 'swallowed'

  with pytest.raises(NodeInterruptedError):
    node_body_that_swallows_errors()
