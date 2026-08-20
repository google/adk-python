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

from google.adk.evaluation.simulation import user_simulator as user_simulator_module
from google.adk.evaluation.simulation.user_simulator import BaseUserSimulatorConfig
from google.adk.evaluation.simulation.user_simulator import NextUserMessage
from google.adk.evaluation.simulation.user_simulator import register_user_simulator
from google.adk.evaluation.simulation.user_simulator import Status
from google.adk.evaluation.simulation.user_simulator import UserSimulator
from google.genai.types import Content
from pydantic import Field
import pytest
from typing_extensions import Literal


def test_next_user_message_validation():
  """Tests post-init validation of NextUserMessage."""
  with pytest.raises(
      ValueError,
      match=(
          "A user_message should be provided if and only if the status is"
          " SUCCESS"
      ),
  ):
    NextUserMessage(status=Status.SUCCESS)

  with pytest.raises(
      ValueError,
      match=(
          "A user_message should be provided if and only if the status is"
          " SUCCESS"
      ),
  ):
    NextUserMessage(status=Status.TURN_LIMIT_REACHED, user_message=Content())

  # these two should not cause exceptions
  NextUserMessage(status=Status.SUCCESS, user_message=Content())
  NextUserMessage(status=Status.TURN_LIMIT_REACHED)


# -----------------------------------------------------------------------------
# `register_user_simulator` -- the extension-point API in `user_simulator.py`.
# End-to-end dispatch through `UserSimulatorProvider` is covered separately in
# `test_user_simulator_provider.py`.
# -----------------------------------------------------------------------------


class _FakeConfig(BaseUserSimulatorConfig):
  """Test-only config subclass with a unique Literal discriminator."""

  type: Literal["fake_sim"] = Field(default="fake_sim")


class _FakeSimulator(UserSimulator):
  """Test-only simulator; internals do not matter, only its type identity."""


def test_register_user_simulator_writes_to_shared_registry():
  """`register_user_simulator(config_type, simulator_type)` must write the

  mapping into `_SIMULATOR_BY_CONFIG_TYPE` so that any consumer -- including
  `UserSimulatorProvider` in another module -- can look it up.
  """
  try:
    register_user_simulator(_FakeConfig, _FakeSimulator)
    assert (
        user_simulator_module._SIMULATOR_BY_CONFIG_TYPE.get(_FakeConfig)
        is _FakeSimulator
    )
  finally:
    # Clean up so we don't leak state into other tests.
    user_simulator_module._SIMULATOR_BY_CONFIG_TYPE.pop(_FakeConfig, None)


def test_register_user_simulator_overwrites_existing_entry():
  """Re-registering the same config type must overwrite the previous entry.

  This lets a test or an out-of-tree extension swap in an alternative
  implementation for the same config type without having to unregister first.
  """

  class _AlternativeFakeSimulator(UserSimulator):
    pass

  try:
    register_user_simulator(_FakeConfig, _FakeSimulator)
    register_user_simulator(_FakeConfig, _AlternativeFakeSimulator)
    assert (
        user_simulator_module._SIMULATOR_BY_CONFIG_TYPE.get(_FakeConfig)
        is _AlternativeFakeSimulator
    )
  finally:
    user_simulator_module._SIMULATOR_BY_CONFIG_TYPE.pop(_FakeConfig, None)
