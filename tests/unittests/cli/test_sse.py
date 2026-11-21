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

"""Tests for the public SSE framing helpers behind `/run_sse`."""

import json

from google.adk.cli import format_event_frames
from google.adk.cli import serialize_event
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types


def _text_event(text: str = "LLM reply") -> Event:
  return Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(role="model", parts=[types.Part(text=text)]),
  )


def _artifact_event() -> Event:
  return Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(role="model", parts=[types.Part(text="LLM reply")]),
      actions=EventActions(artifact_delta={"artifact.txt": 0}),
  )


def test_serialize_event_uses_wire_aliases_and_drops_none_fields():
  """The payload matches the JSON `Event.model_dump` shape clients parse."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name="tool", args={"a": 1})
              )
          ],
      ),
  )

  payload = json.loads(serialize_event(event))

  assert payload["content"]["parts"][0]["functionCall"]["name"] == "tool"
  # snake_case aliases never reach the wire, and None fields are omitted.
  assert "function_call" not in payload["content"]["parts"][0]
  assert "errorCode" not in payload


def test_serialize_event_redacts_credential_secrets():
  """OAuth2 client secrets and PKCE verifiers never reach the frame."""
  auth_config = {
      "authScheme": {
          "type": "oauth2",
          "flows": {
              "authorizationCode": {
                  "authorizationUrl": "https://idp.example.com/auth",
                  "tokenUrl": "https://idp.example.com/token",
              }
          },
      },
      "rawAuthCredential": {
          "authType": "oauth2",
          "oauth2": {
              "clientId": "public-client-id",
              "clientSecret": "should-never-reach-the-client",
          },
      },
  }
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      actions=EventActions(requested_auth_configs={"fc-id": auth_config}),
  )

  frame = serialize_event(event)

  assert "should-never-reach-the-client" not in frame
  payload = json.loads(frame)
  raw_credential = payload["actions"]["requestedAuthConfigs"]["fc-id"][
      "rawAuthCredential"
  ]["oauth2"]
  assert raw_credential["clientId"] == "public-client-id"
  assert "clientSecret" not in raw_credential


def test_serialize_event_drops_internal_metadata():
  """Internal-only custom metadata keys are not exposed to clients."""
  event = _text_event()
  event.custom_metadata = {
      f"{INTERNAL_METADATA_PREFIX}internal": "secret",
      "public": "kept",
  }

  payload = json.loads(serialize_event(event))

  assert payload["customMetadata"] == {"public": "kept"}


def test_format_event_frames_splits_artifact_delta():
  """An event with content and an artifactDelta becomes two frames."""
  frames = format_event_frames(_artifact_event())

  assert len(frames) == 2
  assert all(frame.startswith("data: ") for frame in frames)
  assert all(frame.endswith("\n\n") for frame in frames)

  content_payload, artifact_payload = (
      json.loads(frame.removeprefix("data: ").removesuffix("\n\n"))
      for frame in frames
  )
  assert content_payload["content"]["parts"][0]["text"] == "LLM reply"
  assert content_payload["actions"]["artifactDelta"] == {}
  assert "content" not in artifact_payload
  assert artifact_payload["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_format_event_frames_emits_one_frame_without_split():
  """Callers that must not split, e.g. a function-call resume, get one frame."""
  frames = format_event_frames(_artifact_event(), split_artifact_delta=False)

  assert len(frames) == 1
  payload = json.loads(frames[0].removeprefix("data: ").removesuffix("\n\n"))
  assert payload["content"]["parts"][0]["text"] == "LLM reply"
  assert payload["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_format_event_frames_does_not_split_contentless_events():
  """An artifactDelta without content stays in a single frame."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      actions=EventActions(artifact_delta={"artifact.txt": 0}),
  )

  frames = format_event_frames(event)

  assert len(frames) == 1
  payload = json.loads(frames[0].removeprefix("data: ").removesuffix("\n\n"))
  assert "content" not in payload
  assert payload["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_format_event_frames_does_not_modify_the_event():
  """Formatting leaves the caller's event untouched, including actions."""
  event = _artifact_event()

  format_event_frames(event)

  assert event.actions.artifact_delta == {"artifact.txt": 0}
  assert event.content.parts[0].text == "LLM reply"
