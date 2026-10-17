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

from google.adk.workflow._errors import GraphValidationError
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow._errors import WorkflowConfigurationError
from google.adk.workflow._errors import WorkflowDataError
from google.adk.workflow._errors import WorkflowInvariantError
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


@pytest.mark.parametrize(
    'error_type',
    [GraphValidationError, WorkflowConfigurationError, WorkflowDataError],
)
def test_caller_facing_errors_stay_value_errors(error_type):
  """Naming these errors must not stop existing handlers catching them.

  Each replaced a bare ValueError, so anything already catching ValueError
  around a workflow keeps working.
  """
  with pytest.raises(ValueError):
    raise error_type('boom')


def test_invariant_error_is_not_a_value_error():
  """An engine bug is not the caller's bad input and must not look like it.

  Keeping it off ValueError stops a caller's ``except ValueError`` from
  quietly absorbing a defect in the workflow engine.
  """
  assert issubclass(WorkflowInvariantError, RuntimeError)
  assert not issubclass(WorkflowInvariantError, ValueError)


def test_the_caller_facing_errors_are_distinguishable():
  """The three caller-facing errors say different things, so keep them apart."""
  for error_type in (
      GraphValidationError,
      WorkflowConfigurationError,
      WorkflowDataError,
  ):
    others = {
        GraphValidationError,
        WorkflowConfigurationError,
        WorkflowDataError,
    } - {error_type}
    for other in others:
      assert not issubclass(error_type, other)
