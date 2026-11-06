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

"""Data models for the Street View and Google Maps Platform integration."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class AddressType(StrEnum):
  """Classification of the physical real-estate property."""

  SINGLE_FAMILY = "SINGLE_FAMILY"
  MULTI_UNIT = "MULTI_UNIT"


class PropertyClassification(BaseModel):
  """Result of address normalization and property classification."""

  standardized_address: str = Field(
      ..., description="Full normalized address string."
  )
  base_address: str = Field(
      ..., description="Address without apartment or unit identifier."
  )
  address_type: AddressType = Field(
      ..., description="Single-family vs. multi-unit classification."
  )
  subunit: str | None = Field(
      default=None, description="Extracted subunit (e.g. APT 4B, SUITE 200)."
  )
  location_type: str | None = Field(
      default="RESIDENTIAL", description="RESIDENTIAL or BUSINESS."
  )
  is_missing_subunit: bool = Field(
      default=False,
      description="True if a subpremise is expected but omitted.",
  )
  maps_api_error: str | None = Field(
      default=None,
      description="Diagnostic message if API fallback was triggered.",
  )


class StreetViewResult(BaseModel):
  """Result of a Street View outdoor baseline query."""

  available: bool = Field(
      ..., description="Whether a valid outdoor panoramic photo was found."
  )
  image_bytes: bytes | None = Field(
      default=None, description="JPEG image bytes of the Street View panorama."
  )
  maps_url: str | None = Field(
      default=None,
      description="Direct deep-link URL to Google Maps panorama view.",
  )
  coordinates: tuple[float, float] = Field(
      ..., description="(latitude, longitude) evaluated."
  )
  pano_id: str | None = Field(
      default=None,
      description="Panorama ID from Street View pre-flight metadata.",
  )
  imagery_date: str | None = Field(
      default=None,
      description="Capture date (YYYY-MM) of the Street View panorama.",
  )
  camera_coordinates: tuple[float, float] | None = Field(
      default=None,
      description="Actual (latitude, longitude) of the Street View camera car.",
  )
  heading: float | None = Field(
      default=None,
      description="Compass bearing (0-360 degrees) from camera to target.",
  )


VerdictLabel = Literal[
    "CONFIRMED_MATCH",
    "CONFIRMED_MISMATCH",
    "PLAUSIBLE_ALTERNATE_LOCATION",
    "INSUFFICIENT_VISUAL_CONTEXT",
]


class VisualAuditVerdict(BaseModel):
  """Structured output from Gemini Vision baseline comparison."""

  verdict_label: VerdictLabel = Field(
      ...,
      description="Classification of visual match against reference baseline.",
  )
  match: bool = Field(
      ..., description="True if target photo matches expected location."
  )
  confidence: float = Field(
      ..., description="Confidence score clamped between 0.0 and 1.0."
  )
  reason: str = Field(..., description="Detailed visual rationale.")
  baseline_source: str = Field(
      default="NONE",
      description="HISTORICAL_IMAGERY, STREET_VIEW_FALLBACK, or NONE.",
  )
  street_view_url: str | None = Field(
      default=None,
      description="Deep link to Street View if fallback was used.",
  )
