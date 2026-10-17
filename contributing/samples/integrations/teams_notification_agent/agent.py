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

import os
from typing import Any

from google.adk.agents import Agent
import httpx


def _build_adaptive_card(
    title: str, message: str, facts: dict[str, str] | None
) -> dict[str, Any]:
  body: list[dict[str, Any]] = [
      {
          "type": "TextBlock",
          "text": title,
          "size": "Large",
          "weight": "Bolder",
          "wrap": True,
      },
      {"type": "TextBlock", "text": message, "wrap": True},
  ]
  if facts:
    body.append({
        "type": "FactSet",
        "facts": [{"title": k, "value": v} for k, v in facts.items()],
    })
  return {
      "type": "message",
      "attachments": [{
          "contentType": "application/vnd.microsoft.card.adaptive",
          "contentUrl": None,
          "content": {
              "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
              "type": "AdaptiveCard",
              "version": "1.4",
              "body": body,
          },
      }],
  }


async def send_teams_notification(
    title: str, message: str, facts: dict[str, str] | None = None
) -> dict[str, Any]:
  """Posts a notification card to a Microsoft Teams channel.

  Args:
      title: Short headline for the notification.
      message: Main notification text. Basic Markdown is supported.
      facts: Optional key/value details shown as a list under the message.

  Returns:
      A dict with "status" set to "success" or "error".
  """
  webhook_url = os.environ.get("TEAMS_WEBHOOK_URL")
  if not webhook_url:
    return {"status": "error", "error": "TEAMS_WEBHOOK_URL is not set."}

  payload = _build_adaptive_card(title, message, facts)
  try:
    async with httpx.AsyncClient(timeout=10) as client:
      response = await client.post(webhook_url, json=payload)
  except httpx.HTTPError as e:
    # Don't echo the exception text: it can contain the webhook URL, which
    # embeds a signature that grants post access.
    return {"status": "error", "error": type(e).__name__}

  if not response.is_success:
    return {
        "status": "error",
        "error": f"Teams webhook returned HTTP {response.status_code}.",
    }
  return {"status": "success"}


root_agent = Agent(
    name="teams_notification_agent",
    description="Sends notifications to a Microsoft Teams channel.",
    instruction="""
      You help the user send notifications to a Microsoft Teams channel.
      When the user asks to notify, alert, or post something to Teams, call
      send_teams_notification with a short title, a clear message, and any
      structured details (such as environment, status, or owner) as facts.
      Confirm to the user whether the notification was sent.
    """,
    tools=[send_teams_notification],
)
