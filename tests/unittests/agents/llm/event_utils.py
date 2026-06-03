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

"""Shared test helpers for extracting data from Event lists."""

from __future__ import annotations

from typing import Any

from google.adk.events.event import Event


async def collect_events(node, ctx, node_input=None) -> list[Event]:
  """Collect all events yielded by node.run()."""
  events = []
  async for event in node.run(ctx=ctx, node_input=node_input):
    events.append(event)
  return events


def output_events(events: list[Event]) -> list[Event]:
  """Filter to events that carry output."""
  return [e for e in events if e.output is not None]


def content_events(events: list[Event]) -> list[Event]:
  """Filter to events that have content."""
  return [e for e in events if e.content and e.content.parts]


def text_parts(events: list[Event]) -> list[str]:
  """Extract text strings from content events."""
  return [
      p.text.strip()
      for e in content_events(events)
      for p in e.content.parts
      if p.text
  ]


def function_call_names(events: list[Event]) -> list[str]:
  """Extract function call names from content events."""
  return [
      p.function_call.name
      for e in content_events(events)
      for p in e.content.parts
      if p.function_call
  ]


def function_response_dicts(events: list[Event]) -> list[dict[str, Any]]:
  """Extract function response dicts from events."""
  return [
      dict(p.function_response.response or {})
      for e in content_events(events)
      for p in e.content.parts
      if p.function_response
  ]


def function_responses_by_name(
    events: list[Event],
) -> dict[str, dict[str, Any]]:
  """Extract {tool_name: response_dict} from function response events."""
  return {
      p.function_response.name: dict(p.function_response.response or {})
      for e in content_events(events)
      for p in e.content.parts
      if p.function_response
  }
