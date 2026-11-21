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

"""Sample agent that checks a photo against Street View imagery."""

from __future__ import annotations

from typing import Any
from typing import Literal

from google.adk import Agent
from google.adk.apps.app import App
from google.adk.integrations.google_maps import GoogleMapsToolset
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel
from pydantic import Field

from google import genai

MODEL = "gemini-2.5-flash"

BASELINE_PROMPT = (
    "You are analyzing outdoor reference imagery of a physical address.\n"
    "Describe the property's stable features:\n"
    "- Building type (single-family home, apartment building, townhouse,"
    " commercial)\n"
    "- Exterior materials (brick, vinyl siding, stone, stucco, wood)\n"
    "- Entryway features (porch, steps, door color, columns, awnings,"
    " driveway)\n"
    "- Permanent landmarks or landscaping\n"
    "Ignore weather, vehicles, lighting, seasonal foliage, and transient"
    " objects."
)

AUDIT_PROMPT = (
    "Compare the photo against the reference description and decide whether"
    " the photo was taken at the expected location.\n\n"
    "Address: {address}\n"
    "Property type: {property_type}\n"
    "Street address at the photo's coordinates: {street_context}\n\n"
    "- For SINGLE_FAMILY, expect the porch, front door, garage, or side door"
    " of the home.\n"
    "- For MULTI_UNIT, also accept lobbies, parcel lockers, mailrooms,"
    " leasing offices, or unit doors that fit the building."
)


class Verdict(BaseModel):
  """Result of comparing a photo with the Street View reference."""

  verdict_label: Literal[
      "CONFIRMED_MATCH",
      "CONFIRMED_MISMATCH",
      "PLAUSIBLE_ALTERNATE_LOCATION",
      "INSUFFICIENT_VISUAL_CONTEXT",
  ]
  match: bool
  confidence: float = Field(ge=0.0, le=1.0)
  reason: str


async def compare_photo_to_street_view(
    photo_filename: str, tool_context: ToolContext
) -> dict[str, Any]:
  """Compares an uploaded photo with the saved Street View image.

  Call get_street_view_image first. Call validate_address and
  reverse_geocode first too, if you have an address and photo coordinates.

  Args:
    photo_filename: Artifact filename of the uploaded photo.
  """
  street_view_name = tool_context.state.get("street_view_artifact_name")
  if not street_view_name:
    return {"error": "No Street View image. Call get_street_view_image first."}
  street_view = await tool_context.load_artifact(street_view_name)
  photo = await tool_context.load_artifact(photo_filename)
  if street_view is None or photo is None:
    return {
        "error": f"Could not load {photo_filename!r} or the Street View image."
    }

  client = genai.Client()
  baseline = await client.aio.models.generate_content(
      model=MODEL, contents=[BASELINE_PROMPT, street_view]
  )
  state = tool_context.state
  prompt = AUDIT_PROMPT.format(
      address=state.get("standardized_address", "Unknown"),
      property_type=(
          "MULTI_UNIT" if state.get("is_multi_unit") else "SINGLE_FAMILY"
      ),
      street_context=state.get("reverse_geocoded_address", "Unknown"),
  )
  response = await client.aio.models.generate_content(
      model=MODEL,
      contents=[
          prompt,
          f"Reference description:\n{baseline.text or ''}",
          photo,
      ],
      config=types.GenerateContentConfig(
          response_mime_type="application/json", response_schema=Verdict
      ),
  )
  if not isinstance(response.parsed, Verdict):
    return {"error": "The model did not return a verdict."}
  verdict = response.parsed.model_dump()
  state["photo_audit_verdict"] = verdict
  return verdict


root_agent = Agent(
    model=MODEL,
    name="photo_audit_agent",
    instruction="""\
You check whether a user's photo was taken at a given address.

1. Ask for the address and the coordinates where the photo was taken if the
   user has not given them.
2. Call validate_address on the address.
3. Call reverse_geocode and get_street_view_image with the coordinates.
4. Call compare_photo_to_street_view with the uploaded photo's filename.
5. Report the verdict and the reason. Say plainly when a tool failed.
""",
    tools=[GoogleMapsToolset(), compare_photo_to_street_view],
)

app = App(
    name="google_maps_photo_audit",
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)
