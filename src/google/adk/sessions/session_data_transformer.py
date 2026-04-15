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
from typing import Mapping
from typing import Protocol

from google.adk.events.event import Event


class SessionDataTransformer(Protocol):
  """Hook protocol for selectively transforming DB session records before persist/load.

  This is useful for implementing field-level encryption, PII masking, or secret
  scrubbing at the storage boundary without modifying the in-memory core structures,
  as long as the transformation yields valid storage dictionaries and Events.
  """

  def before_persist_event(self, event: Event) -> Event:
    """Invoked just before serializing and persisting an Event to the database."""
    ...

  def after_load_event(self, event: Event) -> Event:
    """Invoked immediately after loading and deserializing an Event from the database."""
    ...

  def before_persist_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
    """Invoked before persisting state changes (can be full state or partial deltas)."""
    ...

  def after_load_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
    """Invoked after loading a combined application/user/session state dict."""
    ...
