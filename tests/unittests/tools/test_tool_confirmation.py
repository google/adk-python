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

from google.adk.tools.tool_confirmation import ToolConfirmation
from pydantic import ValidationError
import pytest

# ToolConfirmation is gated behind an experimental feature flag, which emits a
# UserWarning on use; that is expected and not under test here.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestToolConfirmation:

  def test_defaults(self):
    confirmation = ToolConfirmation()

    assert confirmation.hint == ""
    assert confirmation.confirmed is False
    assert confirmation.payload is None

  def test_stores_custom_values(self):
    confirmation = ToolConfirmation(
        hint="please confirm", confirmed=True, payload={"amount": 10}
    )

    assert confirmation.hint == "please confirm"
    assert confirmation.confirmed is True
    assert confirmation.payload == {"amount": 10}

  def test_payload_accepts_arbitrary_json_serializable_values(self):
    assert ToolConfirmation(payload=[1, 2, 3]).payload == [1, 2, 3]
    assert ToolConfirmation(payload="raw").payload == "raw"

  def test_forbids_extra_fields(self):
    with pytest.raises(ValidationError):
      ToolConfirmation(unexpected="value")

  def test_round_trips_through_dict(self):
    original = ToolConfirmation(
        hint="confirm transfer", confirmed=True, payload={"to": "bob"}
    )

    assert ToolConfirmation.model_validate(original.model_dump()) == original
