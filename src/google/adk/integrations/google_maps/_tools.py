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

"""Implementations of the Google Maps tools."""

from __future__ import annotations

import base64
import inspect
import logging

from google.genai import types

from ...tools.tool_context import ToolContext
from ._config import GoogleMapsConfig
from ._models import AddressType
from ._services import classify_property_address_async
from ._services import fetch_street_view_baseline
from ._services import MapsApiError
from ._services import reverse_geocode_coordinates_async
from ._services import validate_coordinates

logger = logging.getLogger("google_adk." + __name__)

STREET_VIEW_ARTIFACT_FILENAME = "street_view.jpg"


def _invalid_coordinates(latitude: float, longitude: float) -> str:
  return (
      f"Invalid coordinates ({latitude}, {longitude}). "
      "Latitude must be in [-90, 90] and longitude in [-180, 180]."
  )


async def _persist_street_view_image(
    tool_context: ToolContext | None,
    image_bytes: bytes,
    artifact_filename: str = STREET_VIEW_ARTIFACT_FILENAME,
) -> str | None:
  """Saves Street View JPEG bytes as an ADK Artifact, else as base64 state."""
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
    else:
      tool_context.state["street_view_image_b64"] = base64.b64encode(
          image_bytes
      ).decode("ascii")

  return saved_artifact_name


async def get_street_view_image(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Fetches an outdoor Street View image and saves it to the session."""
  if not validate_coordinates(latitude, longitude):
    return _invalid_coordinates(latitude, longitude)

  try:
    result = await fetch_street_view_baseline(
        latitude, longitude, config=config
    )
  except MapsApiError as exc:
    return f"Street View lookup failed: {exc}."

  if not result.available or not result.image_bytes:
    if tool_context and hasattr(tool_context, "state"):
      tool_context.state["street_view_available"] = False
    return (
        "Street View imagery is not available for coordinates "
        f"({latitude}, {longitude})."
    )

  artifact_name = await _persist_street_view_image(
      tool_context, result.image_bytes
  )

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["street_view_available"] = True
    tool_context.state["street_view_url"] = result.maps_url
    tool_context.state["street_view_pano_id"] = result.pano_id
    tool_context.state["street_view_imagery_date"] = result.imagery_date
    tool_context.state["street_view_heading"] = result.heading

  details = [
      f"Captured Street View imagery for ({latitude}, {longitude}).",
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


async def validate_address(
    raw_address: str,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Validates an address and saves the classification to the session."""
  if not raw_address or not raw_address.strip():
    return "Cannot validate an empty address."

  try:
    classification = await classify_property_address_async(
        raw_address, config=config
    )
  except MapsApiError as exc:
    return f"Address validation failed: {exc}."

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
      "### Address Validation",
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

  return "\n".join(report)


async def reverse_geocode(
    latitude: float,
    longitude: float,
    tool_context: ToolContext,
    config: GoogleMapsConfig | None = None,
) -> str:
  """Resolves coordinates into a street address and saves it to the session."""
  if not validate_coordinates(latitude, longitude):
    return _invalid_coordinates(latitude, longitude)

  try:
    formatted_address = await reverse_geocode_coordinates_async(
        latitude, longitude, config=config
    )
  except MapsApiError as exc:
    return f"Reverse geocoding failed: {exc}."

  if not formatted_address:
    return (
        "Could not resolve a street address for coordinates "
        f"({latitude}, {longitude})."
    )

  if tool_context and hasattr(tool_context, "state"):
    tool_context.state["reverse_geocoded_address"] = formatted_address
    tool_context.state["reverse_geocoded_coordinates"] = [latitude, longitude]

  return (
      "### Reverse Geocoded Address\n"
      f"- **Coordinates**: ({latitude}, {longitude})\n"
      f"- **Street Address**: {formatted_address}"
  )
