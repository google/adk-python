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

"""Tests for the ``@experimental_telemetry`` decorator."""

from __future__ import annotations

import logging
from unittest import mock

from google.adk.telemetry._decorators import experimental_telemetry
import pytest


class _FakeConfig:
  """Stands in for ``TelemetryConfig``, recording what the gate asked about."""

  def __init__(self, *enabled: str, blanket: bool = False):
    self._enabled = frozenset(enabled)
    self.should_emit_experimental_telemetry = blanket
    self.asked: list[str] = []

  def _experimental_feature_enabled(self, feature_name: str) -> bool:
    self.asked.append(feature_name)
    return (
        feature_name in self._enabled or self.should_emit_experimental_telemetry
    )


def test_gated_function_runs_when_its_feature_is_enabled():
  """The wrapped function runs, and the config is not passed on to it."""
  emit = mock.Mock()
  gated = experimental_telemetry(gate="skills")(emit)

  gated(_FakeConfig("skills"), "root_agent", count=3)

  emit.assert_called_once_with("root_agent", count=3)


def test_gated_function_does_not_run_when_its_feature_is_disabled():
  emit = mock.Mock()
  gated = experimental_telemetry(gate="skills")(emit)

  gated(_FakeConfig(), "root_agent", count=3)

  emit.assert_not_called()


def test_the_gate_reads_the_config_it_was_handed():
  """Config doesn't leak between calls."""
  emit = mock.Mock()
  gated = experimental_telemetry(gate="skills")(emit)

  gated(_FakeConfig("skills"))
  gated(_FakeConfig())

  assert emit.call_count == 1


@pytest.mark.parametrize(
    "enabled,expected_calls",
    [
        pytest.param((), 0, id="neither"),
        pytest.param(("skills",), 0, id="only_skills"),
        pytest.param(("workflow",), 0, id="only_workflow"),
        pytest.param(("skills", "workflow"), 1, id="both"),
    ],
)
def test_a_multi_feature_gate_needs_every_feature(
    enabled: tuple[str, ...], expected_calls: int
):
  """Several gates are an AND, not an OR."""
  emit = mock.Mock()
  gated = experimental_telemetry(gate=["skills", "workflow"])(emit)

  gated(_FakeConfig(*enabled))

  assert emit.call_count == expected_calls


def test_the_blanket_switch_runs_an_emitter_whose_features_are_off():
  """``ADK_EXPERIMENTAL_TELEMETRY`` covers every feature, named or not."""
  emit = mock.Mock()
  gated = experimental_telemetry(gate=["skills", "workflow"])(emit)

  gated(_FakeConfig(blanket=True))

  emit.assert_called_once_with()


def test_an_ungated_emitter_rides_on_the_blanket_switch_alone():
  """An empty gate names no feature, so only the master switch decides."""
  on = mock.Mock()
  off = mock.Mock()
  experimental_telemetry(gate=[])(on)(_FakeConfig(blanket=True))
  experimental_telemetry(gate=[])(off)(_FakeConfig("skills"))

  on.assert_called_once_with()
  off.assert_not_called()


def test_a_single_gate_name_is_accepted_as_a_bare_string():
  """``gate='skills'`` and ``gate=['skills']`` mean the same thing."""
  from_string = mock.Mock()
  from_list = mock.Mock()
  experimental_telemetry(gate="skills")(from_string)(_FakeConfig("skills"))
  experimental_telemetry(gate=["skills"])(from_list)(_FakeConfig("skills"))

  from_string.assert_called_once_with()
  from_list.assert_called_once_with()


def test_the_gate_asks_only_about_the_features_it_declares():
  config = _FakeConfig("skills", "workflow")
  experimental_telemetry(gate=["skills", "workflow"])(mock.Mock())(config)

  assert set(config.asked) == {"skills", "workflow"}


def test_a_disabled_gate_says_so_at_debug():
  """A suppressed function can be distinguished from a broken one."""
  func = mock.Mock()
  gated = experimental_telemetry(gate=["skills", "workflow"])(func)

  with mock.patch.object(logging.Logger, "debug") as log_debug:
    gated(_FakeConfig("skills"))

  log_debug.assert_called_once()
  assert log_debug.call_args.args[1] == repr(func)


def test_the_wrapper_keeps_the_wrapped_function_identity():
  """decorator doesn't change the wrapped function's name or doc."""

  @experimental_telemetry(gate="skills")
  def record_skill_load() -> None:
    """Records one skill load."""

  assert record_skill_load.__name__ == "record_skill_load"
  assert record_skill_load.__doc__ == "Records one skill load."


def test_an_exception_from_the_emitter_is_not_swallowed():
  """Decorator is transparent to exceptions."""

  @experimental_telemetry(gate="skills")
  def emit() -> None:
    raise ValueError("the meter was not initialized")

  with pytest.raises(ValueError, match="the meter was not initialized"):
    emit(_FakeConfig("skills"))
