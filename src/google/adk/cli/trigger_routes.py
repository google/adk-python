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

"""Trigger endpoints for batch and event-driven agent invocations.

Provides /trigger/pubsub and /trigger/eventarc endpoints
that enable ADK agents to process Pub/Sub push messages and Eventarc events
without requiring
pre-created sessions.

Features include:
  - Semaphore-based concurrency control to stay within LLM model quota
  - Automatic retry with exponential backoff on 429 / RESOURCE_EXHAUSTED
  - Transient error detection to signal upstream services to retry
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
from typing import Any
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from google.genai import types
from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from ..utils.context_utils import Aclosing

if TYPE_CHECKING:
  from .adk_web_server import AdkWebServer

logger = logging.getLogger("google_adk." + __name__)

TAG_TRIGGERS = "Triggers"

# ---------------------------------------------------------------------------
# Concurrency & retry defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_CONCURRENT = int(os.environ.get("ADK_TRIGGER_MAX_CONCURRENT", "10"))
"""Maximum concurrent agent invocations across all trigger requests."""

DEFAULT_MAX_RETRIES = int(os.environ.get("ADK_TRIGGER_MAX_RETRIES", "3"))
"""Maximum retry attempts for transient (429) errors per row."""

DEFAULT_RETRY_BASE_DELAY = float(
    os.environ.get("ADK_TRIGGER_RETRY_BASE_DELAY", "1.0")
)
"""Base delay in seconds for exponential backoff."""

DEFAULT_RETRY_MAX_DELAY = float(
    os.environ.get("ADK_TRIGGER_RETRY_MAX_DELAY", "30.0")
)
"""Maximum delay in seconds for exponential backoff."""


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


class TransientError(Exception):
  """A transient or retryable error (e.g., a 429 status code)."""


def _is_transient_error(error: Exception) -> bool:
  """Check if an exception represents a transient rate-limit error.

  Checks both the exception type (for google-api-core exceptions) and
  the error message string as a fallback for wrapped or generic errors.
  """
  # Check google.api_core exception types when available.
  try:
    from google.api_core import exceptions as api_exceptions

    if isinstance(error, api_exceptions.ResourceExhausted):
      return True
    if isinstance(error, api_exceptions.TooManyRequests):
      return True
  except ImportError:
    pass

  err_msg = str(error).lower()
  return (
      "429" in err_msg
      or "resource_exhausted" in err_msg
      or "rate limit" in err_msg
      or "quota" in err_msg
  )


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class PubSubMessage(BaseModel):
  """Inner message payload from a Pub/Sub push subscription."""

  data: Optional[str] = Field(
      default=None, description="Base64-encoded message data."
  )
  attributes: Optional[dict[str, str]] = Field(
      default=None, description="Message attributes."
  )
  messageId: Optional[str] = Field(
      default=None, description="Pub/Sub message ID."
  )
  publishTime: Optional[str] = Field(
      default=None, description="Publish timestamp."
  )


class PubSubTriggerRequest(BaseModel):
  """Pub/Sub push subscription request format.

  See: https://cloud.google.com/pubsub/docs/push#receive_push
  """

  message: PubSubMessage
  subscription: Optional[str] = Field(
      default=None,
      description="Full subscription name (e.g. projects/p/subscriptions/s).",
  )


class EventarcTriggerRequest(BaseModel):
  """Eventarc / CloudEvents request format.

  Eventarc delivers events as CloudEvents over HTTP in two modes:

  1. **Structured content mode** (JSON body): All CloudEvents attributes
     and the event data are in the JSON body.  Used by direct HTTP callers.
  2. **Binary content mode** (Eventarc default): CloudEvents attributes are
     sent as ``ce-*`` HTTP headers, and the body contains only the event
     data — typically a Pub/Sub message wrapper for Pub/Sub-sourced events:
     ``{"message": {"data": "<base64>", ...}, "subscription": "..."}``.

  See: https://cloud.google.com/eventarc/docs/cloudevents
  """

  # In structured mode, ``data`` is always present.
  # In binary mode, the entire body is the data (often a Pub/Sub wrapper).
  data: Optional[dict[str, Any]] = Field(
      default=None, description="Event payload data (structured mode)."
  )
  source: Optional[str] = Field(
      default=None, description="CloudEvents source attribute."
  )
  type: Optional[str] = Field(
      default=None, description="CloudEvents type attribute."
  )
  id: Optional[str] = Field(
      default=None, description="CloudEvents id attribute."
  )
  time: Optional[str] = Field(
      default=None, description="CloudEvents time attribute."
  )
  specversion: Optional[str] = Field(
      default=None, description="CloudEvents specversion attribute."
  )

  # Binary mode: Pub/Sub message wrapper fields.
  message: Optional[PubSubMessage] = Field(
      default=None,
      description=(
          "Pub/Sub message wrapper (binary content mode from Eventarc)."
      ),
  )
  subscription: Optional[str] = Field(
      default=None,
      description=(
          "Pub/Sub subscription name (binary content mode from Eventarc)."
      ),
  )

  model_config = {"extra": "allow"}


class TriggerResponse(BaseModel):
  """Standard response for Pub/Sub and Eventarc triggers."""

  status: Literal["success", "error"] = Field(
      description="Processing status: 'success' or error."
  )


# ---------------------------------------------------------------------------
# Trigger Router
# ---------------------------------------------------------------------------


class TriggerRouter:
  """A router that registers /trigger/* routes on a FastAPI application.

  Each trigger endpoint auto-creates an ephemeral session, runs the agent,
  and returns the result in the format expected by the calling service.

  Features include:
    - Semaphore limits concurrent agent calls (default: 10)
    - Transient errors (429 / RESOURCE_EXHAUSTED) are retried with
      exponential backoff + jitter
  """

  DEFAULT_TRIGGER_SOURCES = []
  """Trigger sources registered when ``trigger_sources`` is not specified.
  By default, no triggers are registered to require explicit opt-in via CLI.
  """
  VALID_TRIGGER_SOURCES = ["pubsub", "eventarc"]
  """All trigger sources supported by this router."""

  def __init__(
      self,
      adk_web_server: "AdkWebServer",
      *,
      trigger_sources: Optional[list[str]] = None,
      max_concurrent: int = DEFAULT_MAX_CONCURRENT,
      max_retries: int = DEFAULT_MAX_RETRIES,
      retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
      retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
  ):
    self._server = adk_web_server
    resolved_sources = (
        trigger_sources
        if trigger_sources is not None
        else self.DEFAULT_TRIGGER_SOURCES
    )
    unknown = set(resolved_sources) - set(self.VALID_TRIGGER_SOURCES)
    if unknown:
      logger.warning(
          "Unknown trigger source(s) ignored: %s. Valid sources: %s",
          ", ".join(sorted(unknown)),
          ", ".join(self.VALID_TRIGGER_SOURCES),
      )
    self._trigger_sources = [
        s for s in resolved_sources if s in self.VALID_TRIGGER_SOURCES
    ]
    self._semaphore = asyncio.Semaphore(max_concurrent)
    self._max_retries = max_retries
    self._retry_base_delay = retry_base_delay
    self._retry_max_delay = retry_max_delay

  async def _run_agent(
      self,
      *,
      app_name: str,
      user_id: str,
      message_text: str,
      session_id: str,
  ) -> list[Event]:
    """Run the agent with an auto-created ephemeral session.

    Acquires the concurrency semaphore before execution to prevent
    overwhelming the LLM model quota.

    Args:
      app_name: The target application / agent name.
      user_id: Identifier for observability (derived from trigger metadata).
      message_text: The text input to send to the agent.
      session_id: The session ID to use.

    Returns:
      List of events produced by the agent invocation.
    """
    async with self._semaphore:

      runner = await self._server.get_runner_async(app_name)

      session = await self._server.session_service.get_session(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
      )
      if not session:
        session = await self._server.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

      new_message = types.Content(
          role="user",
          parts=[types.Part(text=message_text)],
      )

      events: list[Event] = []
      async with Aclosing(
          runner.run_async(
              user_id=user_id,
              session_id=session.id,
              new_message=new_message,
          )
      ) as agen:
        async for event in agen:
          events.append(event)

      return events

  async def _run_agent_with_retry(
      self,
      *,
      app_name: str,
      user_id: str,
      message_text: str,
  ) -> list[Event]:
    """Run the agent with retry on transient errors.

    Uses exponential backoff with jitter to handle 429 rate-limit errors.
    After max_retries exhausted, raises TransientError to signal the
    upstream service (Pub/Sub, Eventarc) to retry at a higher level.

    Args:
      app_name: The target application / agent name.
      user_id: Identifier for observability.
      message_text: The text input to send to the agent.

    Returns:
      List of events produced by the agent invocation.

    Raises:
      TransientError: When retries are exhausted on a transient error.
      Exception: For non-transient errors, re-raised immediately.
    """
    last_error: Optional[Exception] = None
    session_id = str(uuid.uuid4())

    for attempt in range(self._max_retries + 1):
      try:
        return await self._run_agent(
            app_name=app_name,
            user_id=user_id,
            message_text=message_text,
            session_id=session_id,
        )
      except Exception as e:
        if not _is_transient_error(e):
          raise

        last_error = e
        if attempt < self._max_retries:
          # Exponential backoff with jitter
          delay = min(
              self._retry_base_delay * (2**attempt),
              self._retry_max_delay,
          )
          jitter = random.uniform(0, delay * 0.5)
          total_delay = delay + jitter
          logger.warning(
              "Transient error (attempt %d/%d), retrying in %.1fs: %s",
              attempt + 1,
              self._max_retries + 1,
              total_delay,
              e,
          )
          await asyncio.sleep(total_delay)
        else:
          logger.exception(
              "Transient error persisted after %d attempts: %s",
              self._max_retries + 1,
              e,
          )

    raise TransientError(
        f"Rate limit exceeded after {self._max_retries + 1} attempts:"
        f" {last_error}"
    )

  def register(self, app: FastAPI) -> None:
    """Register /trigger/* routes on the FastAPI app.

    Only endpoints whose source name appears in ``self._trigger_sources``
    are registered.
    """

    if "pubsub" in self._trigger_sources:

      @app.post(
          "/apps/{app_name}/trigger/pubsub",
          response_model=TriggerResponse,
          tags=[TAG_TRIGGERS],
          summary="Pub/Sub push subscription trigger",
          description=(
              "Processes a message from a Pub/Sub push subscription."
              " Returns 200 on success; errors trigger Pub/Sub retry."
              " Includes automatic retry with backoff on 429 errors."
          ),
      )
      async def trigger_pubsub(
          app_name: str, req: PubSubTriggerRequest, request: Request
      ) -> TriggerResponse:
        user_id = req.subscription or "pubsub-caller"

        decoded_data = None
        data_payload = None
        if req.message.data:
          try:
            decoded_data = base64.b64decode(req.message.data).decode("utf-8")
            try:
              data_payload = json.loads(decoded_data)
            except json.JSONDecodeError:
              data_payload = decoded_data
          except Exception as e:
            logger.exception("Failed to decode Pub/Sub message data")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 message data: {e}",
            ) from e

        message_text = json.dumps(
            {"data": data_payload, "attributes": req.message.attributes or {}}
        )

        logger.info(
            "Pub/Sub trigger: subscription=%s, messageId=%s",
            req.subscription,
            req.message.messageId,
        )

        try:
          await self._run_agent_with_retry(
              app_name=app_name,
              user_id=user_id,
              message_text=message_text,
          )
        except TransientError as te:
          logger.exception("Pub/Sub: transient error after retries: %s", te)
          raise HTTPException(
              status_code=500,
              detail=f"Rate limit exceeded (429). Retryable. {te}",
          ) from te
        except Exception as e:
          logger.exception("Error processing Pub/Sub message: %s", e)
          raise HTTPException(
              status_code=500,
              detail=f"Agent processing failed: {e}",
          ) from e

        return TriggerResponse(status="success")

    if "eventarc" in self._trigger_sources:

      @app.post(
          "/apps/{app_name}/trigger/eventarc",
          response_model=TriggerResponse,
          tags=[TAG_TRIGGERS],
          summary="Eventarc / CloudEvents trigger",
          description=(
              "Processes a CloudEvent delivered by Eventarc."
              " Returns 200 on success; errors trigger Eventarc retry."
              " Includes automatic retry with backoff on 429 errors."
          ),
      )
      async def trigger_eventarc(
          app_name: str, req: EventarcTriggerRequest, request: Request
      ) -> TriggerResponse:

        user_id = (
            req.source or request.headers.get("ce-source") or "eventarc-caller"
        )

        logger.info(
            "Eventarc trigger: source=%s, type=%s, id=%s",
            user_id,
            req.type or request.headers.get("ce-type"),
            req.id or request.headers.get("ce-id"),
        )

        # Extract message text — support both structured and binary modes.
        if req.message:
          # Binary content mode (Eventarc default): body is a Pub/Sub
          # message wrapper with base64-encoded data.
          data_payload = None
          if req.message.data:
            try:
              decoded_data = base64.b64decode(req.message.data).decode("utf-8")
              try:
                data_payload = json.loads(decoded_data)
              except json.JSONDecodeError:
                data_payload = decoded_data
            except Exception:
              data_payload = req.message.data

          message_text = json.dumps(
              {"data": data_payload, "attributes": req.message.attributes or {}}
          )
        elif req.data is not None:
          # Structured content mode: ``data`` dict in body.
          if (
              isinstance(req.data, dict)
              and "message" in req.data
              and isinstance(req.data["message"], dict)
              and "data" in req.data["message"]
          ):
            try:
              decoded_data = base64.b64decode(
                  req.data["message"]["data"]
              ).decode("utf-8")
              try:
                data_payload = json.loads(decoded_data)
              except json.JSONDecodeError:
                data_payload = decoded_data
            except Exception:
              data_payload = req.data["message"]["data"]

            message_text = json.dumps({
                "data": data_payload,
                "attributes": req.data["message"].get("attributes") or {},
            })
          else:
            # Direct CloudEvent
            message_text = json.dumps({
                "data": req.data,
                "attributes": {
                    "ce-id": req.id or request.headers.get("ce-id"),
                    "ce-type": req.type or request.headers.get("ce-type"),
                    "ce-source": req.source or request.headers.get("ce-source"),
                    "ce-specversion": (
                        req.specversion or request.headers.get("ce-specversion")
                    ),
                },
            })
        else:
          # Fallback: serialize whatever we got.
          message_text = json.dumps({
              "data": req.model_dump(exclude_unset=True),
              "attributes": {
                  "ce-id": req.id or request.headers.get("ce-id"),
                  "ce-type": req.type or request.headers.get("ce-type"),
                  "ce-source": req.source or request.headers.get("ce-source"),
                  "ce-specversion": (
                      req.specversion or request.headers.get("ce-specversion")
                  ),
              },
          })

        try:
          await self._run_agent_with_retry(
              app_name=app_name,
              user_id=user_id,
              message_text=message_text,
          )
        except TransientError as te:
          logger.exception("Eventarc: transient error after retries: %s", te)
          raise HTTPException(
              status_code=500,
              detail=f"Rate limit exceeded (429). Retryable. {te}",
          ) from te
        except Exception as e:
          logger.exception("Error processing Eventarc event: %s", e)
          raise HTTPException(
              status_code=500,
              detail=f"Agent processing failed: {e}",
          ) from e

        return TriggerResponse(status="success")
