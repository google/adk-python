import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

from app.models.streaming_request import OptimizationLevel
from google.adk.events.event import Event

logger = logging.getLogger(__name__)


class SSEEventMapper:
  """
  Maps and optimizes ADK Events into Server-Sent Event (SSE) messages.
  """

  def map_event_to_sse_message(
      self, event: Event, optimization_level: OptimizationLevel
  ) -> Optional[str]:
    try:
      if optimization_level == OptimizationLevel.MINIMAL:
        payload = self._create_minimal_payload(event)
      elif optimization_level == OptimizationLevel.BALANCED:
        payload = self._create_balanced_payload(event)
      else:
        payload = self._create_full_compat_payload(event)

      if payload is None:
        return None

      sse_json = json.dumps(payload, separators=(",", ":"))
      return f"data: {sse_json}\n\n"

    except (TypeError, AttributeError) as e:
      logger.error(f"Failed to map event: {e}", exc_info=True)
      return f"data: {event.model_dump_json(exclude_none=True)}\n\n"

  def _create_minimal_payload(self, event: Event) -> Optional[Dict[str, Any]]:
    # Skip debug or system events based on author
    if event.author in ["system", "debug"]:
      return None

    payload = {"author": event.author}

    # Extract text content if available
    if event.content and event.content.parts:
      text_parts = [
          part.text
          for part in event.content.parts
          if hasattr(part, "text") and part.text
      ]
      if text_parts:
        payload["text"] = " ".join(text_parts)

    return payload

  def _create_balanced_payload(self, event: Event) -> Optional[Dict[str, Any]]:
    payload = self._create_minimal_payload(event)
    if payload is None:
      return None
    payload["invocation_id"] = event.invocation_id
    return payload

  def _create_full_compat_payload(self, event: Event) -> Dict[str, Any]:
    return event.model_dump(exclude_none=True)
