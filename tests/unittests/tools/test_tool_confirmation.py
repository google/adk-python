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

"""Tests for ToolConfirmation.

Verifies that ToolConfirmation correctly stores and validates confirmation
states.
"""

from __future__ import annotations

from google.adk.tools.tool_confirmation import ToolConfirmation
from pydantic import ValidationError
import pytest

# ToolConfirmation is gated behind an experimental feature flag, which emits a
# UserWarning on use; that is expected and not under test here.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestToolConfirmation:
  """Tests for the ToolConfirmation model."""

  def test_default_values_are_empty(self):
    """A default ToolConfirmation has empty hint, unconfirmed, and no payload."""
    confirmation = ToolConfirmation()

    assert confirmation.hint == ""
    assert confirmation.confirmed is False
    assert confirmation.payload is None

  def test_initialization_retains_provided_values(self):
    """ToolConfirmation stores values provided during initialization."""
    confirmation = ToolConfirmation(
        hint="please confirm", confirmed=True, payload={"amount": 10}
    )

    assert confirmation.hint == "please confirm"
    assert confirmation.confirmed is True
    assert confirmation.payload == {"amount": 10}

  @pytest.mark.parametrize(
      "payload_value",
      [
          [1, 2, 3],
          "raw",
      ],
  )
  def test_payload_accepts_json_serializable_values(self, payload_value):
    """ToolConfirmation payload accepts various JSON-serializable values."""
    confirmation = ToolConfirmation(payload=payload_value)

    assert confirmation.payload == payload_value

  @pytest.mark.parametrize(
      "payload_value",
      [
          lambda x: x,
          object(),
      ],
  )
  def test_payload_accepts_non_json_serializable_values(self, payload_value):
    """ToolConfirmation payload accepts non-JSON-serializable values."""
    confirmation = ToolConfirmation(payload=payload_value)

    assert confirmation.payload == payload_value

  def test_initialization_fails_with_extra_fields(self):
    """ToolConfirmation forbids extra fields during initialization."""
    with pytest.raises(ValidationError):
      ToolConfirmation(unexpected="value")

  def test_serialization_round_trip_preserves_equality(self):
    """ToolConfirmation can be serialized and deserialized back to its original state."""
    original = ToolConfirmation(
        hint="confirm transfer", confirmed=True, payload={"to": "bob"}
    )

    dumped = original.model_dump()
    validated = ToolConfirmation.model_validate(dumped)

    assert validated == original
