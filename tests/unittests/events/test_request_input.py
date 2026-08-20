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

"""Unit tests for the RequestInput event model."""

from __future__ import annotations

import uuid

from google.adk.events.request_input import RequestInput
from google.adk.platform import uuid as platform_uuid


class TestRequestInputInterruptId:

  def teardown_method(self) -> None:
    platform_uuid.reset_id_provider()

  def test_default_interrupt_id_is_a_uuid(self):
    """Without a custom provider, the default interrupt_id is a uuid."""
    request = RequestInput()

    # Should be parseable as uuid
    uuid.UUID(request.interrupt_id)

  def test_default_interrupt_id_uses_platform_id_provider(self):
    """The default interrupt_id is minted via the platform uuid seam.

    Frameworks that replay agent workflows (e.g. durable execution engines)
    install a deterministic id provider; the generated interrupt_id must be
    stable across replays because recorded user responses reference it.
    """
    platform_uuid.set_id_provider(lambda: "deterministic-id")

    request = RequestInput()

    assert request.interrupt_id == "deterministic-id"

  def test_explicit_interrupt_id_is_preserved(self):
    """An explicitly provided interrupt_id bypasses the provider."""
    platform_uuid.set_id_provider(lambda: "deterministic-id")

    request = RequestInput(interrupt_id="explicit-id")

    assert request.interrupt_id == "explicit-id"
