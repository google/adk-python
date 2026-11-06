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

"""ADK Tool interfaces and Gemini Vision baseline comparison utilities."""

from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import re
from typing import Any

from google.genai import types

from ...agents.llm_agent import LlmAgent
from ...tools.tool_context import ToolContext
from ._config import default_config
from ._config import GoogleMapsConfig
from ._models import AddressType
from ._models import VisualAuditVerdict
from ._services import classify_property_address_async
from ._services import fetch_street_view_baseline
from ._services import reverse_geocode_coordinates_async
from ._services import validate_coordinates

logger = logging.getLogger("google_adk." + __name__)

STREET_VIEW_ARTIFACT_FILENAME = "street_view_baseline.jpg"
_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE
)

BASELINE_SYNTHESIS_PROMPT = (
    "You are analyzing outdoor reference imagery of a physical address.\n"
    "Synthesize a concise reference description of the property's STABLE"
    " features:\n"
    "- Architectural building type (single-family home, apartment building,"
    " townhouse, commercial)\n"
    "- Exterior materials (brick, vinyl siding, stone, stucco, wood)\n"
    "- Primary entryway visual features (porch, steps, door color, columns,"
    " awnings, driveway alignment)\n"
    "- Distinguishing permanent landmarks or landscaping\n"
    "Ignore weather, vehicle positions, lighting, seasonal foliage, and"
    " transient objects."
)

AUDIT_VERDICT_PROMPT = (
    "Compare the target audit photo against the visual baseline reference.\n"
    "Determine whether the target photo matches the expected physical"
    " location.\n\n"
    "Address Context: {address}\n"
    "Property Type: {property_type}\n"
    "Drop-off Street Context (Reverse Geocoded): {street_context}\n\n"
    "Decision Boundary Guidance:\n"
    "- If Property Type is SINGLE_FAMILY: Expect tight visual alignment with"
    " the primary porch, front entryway, garage, or side door of the home.\n"
    "- If Property Type is MULTI_UNIT: Allow for distributed drop-off"
    " locations such as building lobbies, parcel lockers, mailrooms, leasing"
    " offices, or unit doors consistent with the complex's architecture.\n\n"
    "Output strict JSON with the following schema:\n"
    "{{\n"
    '  "verdict_label": "CONFIRMED_MATCH | CONFIRMED_MISMATCH |'
    ' PLAUSIBLE_ALTERNATE_LOCATION | INSUFFICIENT_VISUAL_CONTEXT",\n'
    '  "match": true | false,\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reason": "Detailed visual rationale comparing stable features"\n'
    "}}"
)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
  """Safely parses a JSON object from LLM text, stripping markdown fences."""
  cleaned = _CODE_FENCE_RE.sub("", raw_text).strip()
  parsed = json.loads(cleaned)
  if not isinstance(parsed, dict):
    raise ValueError("Expected JSON object from Gemini audit response.")
  return parsed


async def _persist_street_view_image(
    tool_context: ToolContext | None,
    image_bytes: bytes,
    artifact_filename: str = STREET_VIEW_ARTIFACT_FILENAME,
) -> str | None:
  """Saves Street View JPEG bytes as an ADK Artifact plus base64 state."""
  if tool_context is None:
    return None

  saved_artifact_name: str | None = None
  if hasattr(tool_context, "save_artifact"):
    try:
      part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
      res = tool_context.save_artifact(
          filename=artifact_filename, artifact=part
      )
      if inspect.isawaitable(res):
        await res
      saved_artifact_name = artifact_filename
    except Exception as exc:  # pylint: disable=broad-except
      logger.debug(
          "save_artifact unavailable; falling back to base64 state: %s",
          exc,
      )

  if hasattr(tool_context, "state"):
    if saved_artifact_name:
      tool_context.state["street_view_artifact_name"] = saved_artifact_name
    tool_context.state["street_view_image_b64"] = base64.b64encode(
        image_bytes
    ).decode("ascii")

  return saved_artifact_name


async def _execute_street_view_fallback(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Executes Street View pre-flight and panorama capture into session state."""
  if not validate_coordinates(latitude, longitude):
    return (
        f"Invalid GPS coordinates ({latitude}, {longitude}). "
        "Latitude must be in [-90, 90] and longitude in [-180, 180]."
    )

  result = await fetch_street_view_baseline(latitude, longitude, config=config)
  if not result.available or not result.image_bytes:
    if tool_context and hasattr(tool_context, "state"):
      tool_context.state["street_view_baseline_used"] = False
    return (
        "Street View imagery is not available for coordinates "
        f"({latitude}, {longitude})."
    )

  artifact_name = await _persist_street_view_image(
      tool_context, result.image_bytes
  )

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["street_view_baseline_used"] = True
    tool_context.state["street_view_url"] = result.maps_url
    tool_context.state["street_view_pano_id"] = result.pano_id
    tool_context.state["street_view_imagery_date"] = result.imagery_date
    tool_context.state["street_view_heading"] = result.heading

  details = [
      (
          "Successfully captured Street View baseline imagery for "
          f"({latitude}, {longitude})."
      ),
      f"- **Deep Link**: {result.maps_url}",
  ]
  if result.pano_id:
    details.append(f"- **Panorama ID**: `{result.pano_id}`")
  if result.imagery_date:
    details.append(f"- **Capture Date**: {result.imagery_date}")
  if result.heading is not None:
    details.append(f"- **Camera Heading**: {result.heading}°")
  if artifact_name:
    details.append(f"- **ADK Artifact**: `{artifact_name}`")

  return "\n".join(details)


async def _execute_property_classification(
    raw_address: str,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Executes address validation and property boundary classification."""
  if not raw_address or not raw_address.strip():
    return "Cannot classify an empty address string."

  classification = await classify_property_address_async(
      raw_address, config=config
  )

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["address_classification"] = classification.model_dump()
    tool_context.state["is_multi_unit"] = (
        classification.address_type == AddressType.MULTI_UNIT
    )
    tool_context.state["is_missing_subunit"] = classification.is_missing_subunit
    tool_context.state["standardized_address"] = (
        classification.standardized_address
    )

  report = [
      "### Address Property Classification",
      f"- **Standardized Address**: {classification.standardized_address}",
      f"- **Base Address**: {classification.base_address}",
      f"- **Property Type**: {classification.address_type.value}",
      f"- **Location Type**: {classification.location_type}",
  ]
  if classification.subunit:
    report.append(f"- **Sub-unit**: {classification.subunit}")
  if classification.is_missing_subunit:
    report.append(
        "- **Warning**: Address appears to be a multi-unit property "
        "with a missing unit/apartment number."
    )
  if classification.maps_api_error:
    report.append(
        f"- *Note*: Maps API fallback active ({classification.maps_api_error})"
    )

  return "\n".join(report)


async def _execute_reverse_geocode(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Executes reverse geocoding and stores resolved street context in state."""
  if not validate_coordinates(latitude, longitude):
    return (
        f"Invalid GPS coordinates ({latitude}, {longitude}). "
        "Latitude must be in [-90, 90] and longitude in [-180, 180]."
    )

  formatted_address = await reverse_geocode_coordinates_async(
      latitude, longitude, config=config
  )
  if not formatted_address:
    return (
        "Could not resolve a street address for GPS coordinates "
        f"({latitude}, {longitude})."
    )

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["reverse_geocoded_address"] = formatted_address
    tool_context.state["dropoff_coordinates"] = [latitude, longitude]

  return (
      "### Reverse Geocoded Street Context\n"
      f"- **Coordinates**: ({latitude}, {longitude})\n"
      f"- **Resolved Street Address**: {formatted_address}"
  )


async def street_view_fallback_tool(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
) -> str:
  """Queries Google Street View to establish an outdoor visual baseline.

  Use this tool when no historical ground-truth photos exist for an address
  (cold-start). Performs a pre-flight metadata check before downloading imagery
  and saves the resulting panorama into ADK session artifacts and state.

  Args:
      latitude: Target property or drop-off latitude in decimal degrees.
      longitude: Target property or drop-off longitude in decimal degrees.
      tool_context: ADK ToolContext automatically injected by the runner.
  """
  return await _execute_street_view_fallback(
      latitude=latitude,
      longitude=longitude,
      tool_context=tool_context,
      config=default_config,
  )


async def property_classification_tool(
    raw_address: str,
    tool_context: ToolContext,
) -> str:
  """Standardizes an address and classifies SINGLE_FAMILY vs MULTI_UNIT.

  Detects omitted apartment/suite numbers via the Google Address Validation API
  and stores the property classification in ADK session state to set spatial
  decision boundaries.

  Args:
      raw_address: The raw customer or recipient street address to validate.
      tool_context: ADK ToolContext automatically injected by the runner.
  """
  return await _execute_property_classification(
      raw_address=raw_address,
      tool_context=tool_context,
      config=default_config,
  )


async def reverse_geocode_tool(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
) -> str:
  """Resolves GPS telemetry coordinates into a physical street address.

  Saves the resolved street context into ADK session state for comparison
  against the intended recipient address.

  Args:
      latitude: GPS latitude in decimal degrees.
      longitude: GPS longitude in decimal degrees.
      tool_context: ADK ToolContext automatically injected by the runner.
  """
  return await _execute_reverse_geocode(
      latitude=latitude,
      longitude=longitude,
      tool_context=tool_context,
      config=default_config,
  )


async def audit_photo_with_fallback(
    target_photo_bytes: bytes,
    historical_photos: list[bytes] | None,
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
    genai_client: Any,
    model: str | None = None,
    config: GoogleMapsConfig | None = None,
) -> dict[str, Any]:
  """Compares a target photo against historical photos or Street View."""
  resolved_model = model or os.getenv("MODEL_NAME") or LlmAgent.DEFAULT_MODEL

  baseline_parts: list[types.Part] = []
  baseline_source = "NONE"
  sv_url: str | None = None

  if historical_photos:
    for photo in historical_photos:
      baseline_parts.append(
          types.Part.from_bytes(data=photo, mime_type="image/jpeg")
      )
    baseline_source = "HISTORICAL_IMAGERY"
  else:
    sv_result = await fetch_street_view_baseline(
        latitude, longitude, config=config
    )
    if sv_result.available and sv_result.image_bytes:
      baseline_parts.append(
          types.Part.from_bytes(
              data=sv_result.image_bytes, mime_type="image/jpeg"
          )
      )
      baseline_source = "STREET_VIEW_FALLBACK"
      sv_url = sv_result.maps_url
      await _persist_street_view_image(tool_context, sv_result.image_bytes)
      if tool_context and hasattr(tool_context, "state"):
        tool_context.state["street_view_baseline_used"] = True
        tool_context.state["street_view_url"] = sv_result.maps_url
        tool_context.state["street_view_pano_id"] = sv_result.pano_id
        tool_context.state["street_view_imagery_date"] = sv_result.imagery_date

  if not baseline_parts:
    verdict = VisualAuditVerdict(
        verdict_label="INSUFFICIENT_VISUAL_CONTEXT",
        match=False,
        confidence=0.0,
        reason=(
            "No historical baseline photos or outdoor Street View imagery "
            "available for location."
        ),
        baseline_source="NONE",
        street_view_url=None,
    )
    return verdict.model_dump()

  baseline_response = await genai_client.aio.models.generate_content(
      model=resolved_model,
      contents=[BASELINE_SYNTHESIS_PROMPT, *baseline_parts],
  )
  baseline_description = baseline_response.text or ""

  state: Any = (
      tool_context.state
      if (tool_context and hasattr(tool_context, "state"))
      else {}
  )
  address_str = state.get("standardized_address", "Unknown")
  property_type_str = (
      "MULTI_UNIT" if state.get("is_multi_unit") else "SINGLE_FAMILY"
  )
  street_context_str = state.get("reverse_geocoded_address", "Not evaluated")

  audit_response = await genai_client.aio.models.generate_content(
      model=resolved_model,
      contents=[
          AUDIT_VERDICT_PROMPT.format(
              address=address_str,
              property_type=property_type_str,
              street_context=street_context_str,
          ),
          (
              f"Visual Baseline Reference ({baseline_source}):\n"
              f"{baseline_description}"
          ),
          types.Part.from_bytes(
              data=target_photo_bytes, mime_type="image/jpeg"
          ),
      ],
      config=types.GenerateContentConfig(response_mime_type="application/json"),
  )

  parsed_data: dict[str, Any] = {}
  if getattr(audit_response, "parsed", None) and isinstance(
      audit_response.parsed, dict
  ):
    parsed_data = dict(audit_response.parsed)
  elif getattr(audit_response, "text", None):
    parsed_data = _parse_json_object(audit_response.text)

  raw_conf = parsed_data.get("confidence", 0.0)
  try:
    clamped_conf = max(0.0, min(1.0, float(raw_conf)))
  except (TypeError, ValueError):
    clamped_conf = 0.0

  verdict_obj = VisualAuditVerdict(
      verdict_label=parsed_data.get(
          "verdict_label", "INSUFFICIENT_VISUAL_CONTEXT"
      ),
      match=bool(parsed_data.get("match", False)),
      confidence=clamped_conf,
      reason=str(parsed_data.get("reason", "No rationale returned by model.")),
      baseline_source=baseline_source,
      street_view_url=sv_url,
  )
  output = verdict_obj.model_dump()

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["visual_audit_verdict"] = output

  return output
