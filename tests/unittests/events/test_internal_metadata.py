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

"""Tests for the ADK-internal custom_metadata helpers."""

from __future__ import annotations

from unittest import mock

from google.adk.events import _internal_metadata
from google.adk.events._internal_metadata import internal_metadata
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import mark_restored
from google.adk.events._internal_metadata import public_event
from google.adk.events._internal_metadata import public_metadata
from google.adk.events._internal_metadata import public_session
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.events._internal_metadata import without_internal_metadata
from google.adk.events.event import Event
from google.adk.sessions.session import Session
import pytest

_INTERNAL_KEY = INTERNAL_METADATA_PREFIX + "anything"


def test_without_internal_metadata_keeps_other_keys():
  metadata = {"keep": 1, _INTERNAL_KEY: "x", RESTORED_EVENT_KEY: True, 2: "y"}

  assert without_internal_metadata(metadata) == {"keep": 1, 2: "y"}
  assert _INTERNAL_KEY in metadata


def test_without_internal_metadata_logs_dropped_keys():
  with mock.patch.object(_internal_metadata.logger, "debug") as debug:
    without_internal_metadata({"keep": 1, _INTERNAL_KEY: "x"})

  debug.assert_called_once()
  assert _INTERNAL_KEY in str(debug.call_args)
  assert "keep" not in str(debug.call_args)


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"keep": 1, _INTERNAL_KEY: "x", 2: "y"}, {"keep": 1, 2: "y"}),
        ({RESTORED_EVENT_KEY: True}, None),
        ({}, None),
        (None, None),
    ],
    ids=["mixed", "only_internal", "empty", "none"],
)
def test_public_metadata_drops_internal_keys(metadata, expected):
  assert public_metadata(metadata) == expected


def test_public_metadata_does_not_log():
  """Removal on the way out is routine, unlike a caller-supplied key."""
  with mock.patch.object(_internal_metadata.logger, "debug") as debug:
    public_metadata({"keep": 1, _INTERNAL_KEY: "x"})
    public_event(Event(author="user", custom_metadata={_INTERNAL_KEY: "x"}))

  debug.assert_not_called()


def test_without_internal_metadata_passes_none_through():
  assert without_internal_metadata(None) is None


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, {}),
        ({"keep": 1}, {}),
        ({"keep": 1, _INTERNAL_KEY: "x"}, {_INTERNAL_KEY: "x"}),
    ],
    ids=["none", "public", "mixed"],
)
def test_internal_metadata_keeps_only_internal_keys(metadata, expected):
  assert internal_metadata(metadata) == expected


@pytest.mark.parametrize(
    "custom_metadata",
    [
        None,
        {"keep": 1},
        {"keep": 1, _INTERNAL_KEY: "planted", RESTORED_EVENT_KEY: False},
    ],
)
def test_mark_restored_strips_internal_keys_and_sets_marker(custom_metadata):
  event = Event(author="user", custom_metadata=custom_metadata)

  assert mark_restored(event) is event

  expected = {"keep": 1} if custom_metadata else {}
  assert event.custom_metadata == {**expected, RESTORED_EVENT_KEY: True}


@pytest.mark.parametrize(
    "custom_metadata", [None, {}, {"keep": 1}], ids=["none", "empty", "public"]
)
def test_public_event_returns_event_without_internal_keys_unchanged(
    custom_metadata,
):
  event = Event(author="user", custom_metadata=custom_metadata)

  assert public_event(event) is event


@pytest.mark.parametrize(
    "custom_metadata, expected",
    [
        ({"keep": 1, _INTERNAL_KEY: "x"}, {"keep": 1}),
        ({RESTORED_EVENT_KEY: True}, None),
    ],
    ids=["mixed", "only_internal"],
)
def test_public_event_strips_internal_keys_from_a_copy(
    custom_metadata, expected
):
  event = Event(author="user", custom_metadata=custom_metadata)

  public = public_event(event)

  assert public is not event
  assert public.custom_metadata == expected
  assert public.id == event.id
  assert event.custom_metadata == custom_metadata


def test_public_session_strips_every_event_and_keeps_the_original():
  session = Session(id="s", app_name="app", user_id="u")
  session.events = [
      Event(author="user", custom_metadata={RESTORED_EVENT_KEY: True}),
      Event(author="agent", custom_metadata={"keep": 1}),
  ]

  public = public_session(session)

  assert [e.custom_metadata for e in public.events] == [None, {"keep": 1}]
  assert public.events[1] is session.events[1]
  assert session.events[0].custom_metadata == {RESTORED_EVENT_KEY: True}


def test_public_session_returns_session_without_internal_keys_unchanged():
  session = Session(id="s", app_name="app", user_id="u")
  session.events = [Event(author="user", custom_metadata={"keep": 1})]

  assert public_session(session) is session


def test_helpers_pass_non_mapping_metadata_through():
  """Callers that stub events (e.g. with Mock) keep working unchanged."""
  stub = mock.Mock()

  assert without_internal_metadata(stub) is stub
  assert public_metadata(stub) is stub
  assert not internal_metadata(stub)
  event = mock.Mock(spec=Event)
  event.custom_metadata = stub
  assert public_event(event) is event
