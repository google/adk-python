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

"""Unit tests for the Google Street View and Maps Platform integration."""

import inspect
import json
import math
import os
import sys
from typing import Any
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations.street_view import AddressType
from google.adk.integrations.street_view import audit_photo_with_fallback
from google.adk.integrations.street_view import calculate_compass_bearing
from google.adk.integrations.street_view import classify_property_address
from google.adk.integrations.street_view import classify_property_address_async
from google.adk.integrations.street_view import clear_caches
from google.adk.integrations.street_view import fetch_street_view_baseline
from google.adk.integrations.street_view import GoogleMapsConfig
from google.adk.integrations.street_view import property_classification_tool
from google.adk.integrations.street_view import reverse_geocode_coordinates
from google.adk.integrations.street_view import reverse_geocode_coordinates_async
from google.adk.integrations.street_view import reverse_geocode_tool
from google.adk.integrations.street_view import STREET_VIEW_ARTIFACT_FILENAME
from google.adk.integrations.street_view import street_view_fallback_tool
from google.adk.integrations.street_view import StreetViewToolset
from google.adk.integrations.street_view import validate_coordinates
from google.adk.integrations.street_view._services import _BoundedLRUCache


class _DummyToolContext:
  """Lightweight ToolContext test double tracking state and saved artifacts."""

  def __init__(self) -> None:
    self.state: dict[str, Any] = {}
    self.saved_artifacts: dict[str, Any] = {}

  async def save_artifact(
      self,
      filename: str,
      artifact: Any,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    del custom_metadata
    self.saved_artifacts[filename] = artifact
    return 0


class TestCompassBearingAndCoordinates(unittest.TestCase):
  """Tests WGS84 coordinate bounds and compass heading calculation."""

  def test_cardinal_directions(self) -> None:
    self.assertAlmostEqual(
        calculate_compass_bearing(37.0, -122.0, 37.001, -122.0) or 0.0,
        0.0,
        places=1,
    )
    self.assertAlmostEqual(
        calculate_compass_bearing(37.0, -122.0, 37.0, -121.999) or 0.0,
        90.0,
        places=0,
    )
    self.assertAlmostEqual(
        calculate_compass_bearing(37.0, -122.0, 36.999, -122.0) or 0.0,
        180.0,
        places=1,
    )
    self.assertAlmostEqual(
        calculate_compass_bearing(37.0, -122.0, 37.0, -122.001) or 0.0,
        270.0,
        places=0,
    )

  def test_identical_or_invalid_coordinates_returns_none(self) -> None:
    self.assertIsNone(
        calculate_compass_bearing(37.4220, -122.0841, 37.4220, -122.0841)
    )
    self.assertFalse(validate_coordinates(95.0, -122.0))
    self.assertFalse(validate_coordinates(37.0, -190.0))
    self.assertFalse(validate_coordinates(math.nan, -122.0))
    self.assertIsNone(
        calculate_compass_bearing(95.0, -122.0, 37.4220, -122.0841)
    )

  def test_bounded_lru_cache_evicts_least_recently_used(self) -> None:
    cache: _BoundedLRUCache[str, int] = _BoundedLRUCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    self.assertEqual(cache.get("a"), (True, 1))
    cache.put("c", 3)
    self.assertEqual(cache.get("b"), (False, None))
    self.assertEqual(cache.get("a"), (True, 1))
    self.assertEqual(cache.get("c"), (True, 3))


class TestGoogleMapsConfig(unittest.TestCase):
  """Tests API key resolution priority, fingerprinting, and lazy imports."""

  def test_explicit_api_key_priority(self) -> None:
    with patch.dict(os.environ, {"MAPS_API_KEY": "env-key"}, clear=False):
      cfg = GoogleMapsConfig(api_key="explicit-key")
      self.assertEqual(cfg.get_api_key(), "explicit-key")
      self.assertNotEqual(cfg.get_key_fingerprint(), "explicit-key")
      self.assertEqual(len(cfg.get_key_fingerprint()), 16)

  def test_environment_variable_fallback(self) -> None:
    with patch.dict(os.environ, {"MAPS_API_KEY": "env-key-123"}, clear=False):
      cfg = GoogleMapsConfig()
      self.assertEqual(cfg.get_api_key(), "env-key-123")

  def test_unconfigured_or_placeholder_returns_none(self) -> None:
    with patch.dict(
        os.environ,
        {
            "MAPS_API_KEY": "<TODO: update-this-value>",
            "MAPS_API_KEY_SECRET_RESOURCE": "<TODO: update-this-value>",
        },
        clear=True,
    ):
      cfg = GoogleMapsConfig()
      self.assertIsNone(cfg.get_api_key())
      self.assertEqual(cfg.get_key_fingerprint(), "unconfigured")

  def test_missing_secret_manager_extra_raises_helpful_import_error(
      self,
  ) -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.dict(sys.modules, {"google.cloud.secretmanager": None}),
    ):
      cfg = GoogleMapsConfig(
          secret_resource_name="projects/p/secrets/s/versions/latest"
      )
      with self.assertRaisesRegex(ImportError, r"google-adk\[gcp\]"):
        cfg.get_api_key()


class TestServicesAndTools(unittest.IsolatedAsyncioTestCase):
  """Tests Street View pre-flight, Address Validation, Geocoding, and Tools."""

  def setUp(self) -> None:
    super().setUp()
    clear_caches()
    self.config = GoogleMapsConfig(api_key="AIzaSySecretKey12345")

  async def test_street_view_preflight_zero_results_skips_image(self) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def mock_http_get(
        url: str,
        params: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del timeout, headers
      calls.append((url, params))
      return 200, json.dumps({"status": "ZERO_RESULTS"}).encode("utf-8")

    with patch(
        "google.adk.integrations.street_view._services._async_http_get",
        side_effect=mock_http_get,
    ):
      result = await fetch_street_view_baseline(
          37.422, -122.084, config=self.config
      )
      self.assertFalse(result.available)
      self.assertIsNone(result.image_bytes)
      self.assertEqual(len(calls), 1)
      self.assertIn("streetview/metadata", calls[0][0])

  async def test_street_view_preflight_ok_fetches_image_with_heading(
      self,
  ) -> None:
    fake_jpeg = b"\xff\xd8\xff\xe0FAKE_JPEG_DATA"
    calls: list[tuple[str, dict[str, Any]]] = []

    async def mock_http_get(
        url: str,
        params: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del timeout, headers
      calls.append((url, params))
      if "metadata" in url:
        meta = {
            "status": "OK",
            "pano_id": "pano_abc123",
            "date": "2025-08",
            "location": {"lat": 37.4219, "lng": -122.0840},
        }
        return 200, json.dumps(meta).encode("utf-8")
      return 200, fake_jpeg

    with patch(
        "google.adk.integrations.street_view._services._async_http_get",
        side_effect=mock_http_get,
    ):
      result = await fetch_street_view_baseline(
          37.4220, -122.0840, config=self.config
      )
      self.assertTrue(result.available)
      self.assertEqual(result.image_bytes, fake_jpeg)
      self.assertEqual(result.pano_id, "pano_abc123")
      self.assertEqual(result.imagery_date, "2025-08")
      self.assertEqual(result.heading, 0.0)
      self.assertIn("pano=pano_abc123", result.maps_url or "")
      self.assertEqual(len(calls), 2)
      self.assertEqual(calls[1][1].get("pano"), "pano_abc123")
      self.assertEqual(calls[1][1].get("heading"), 0.0)

  async def test_address_validation_uses_header_and_preserves_city_state(
      self,
  ) -> None:
    fake_api_response = {
        "result": {
            "address": {
                "formattedAddress": (
                    "500 Terry A Francois Blvd, San Francisco, CA 94158, USA"
                ),
                "addressComponents": [
                    {
                        "componentType": "street_number",
                        "componentName": {"text": "500"},
                    },
                    {
                        "componentType": "route",
                        "componentName": {"text": "Terry A Francois Blvd"},
                    },
                    {
                        "componentType": "locality",
                        "componentName": {"text": "San Francisco"},
                    },
                    {
                        "componentType": "administrative_area_level_1",
                        "componentName": {"text": "CA"},
                    },
                    {
                        "componentType": "postal_code",
                        "componentName": {"text": "94158"},
                    },
                ],
                "missingComponentTypes": ["subpremise"],
            },
            "metadata": {"business": False},
        }
    }
    captured_headers: list[dict[str, str] | None] = []
    captured_params: list[dict[str, Any]] = []

    async def mock_post(
        url: str,
        params: dict[str, Any],
        payload: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del url, payload, timeout
      captured_params.append(params)
      captured_headers.append(headers)
      return 200, json.dumps(fake_api_response).encode("utf-8")

    with patch(
        "google.adk.integrations.street_view._services._async_http_post_json",
        side_effect=mock_post,
    ):
      res = await classify_property_address_async(
          "500 Terry A Francois Blvd, SF", config=self.config
      )
      self.assertEqual(res.address_type, AddressType.MULTI_UNIT)
      self.assertTrue(res.is_missing_subunit)
      self.assertEqual(
          res.base_address,
          "500 TERRY A FRANCOIS BLVD, SAN FRANCISCO, CA 94158",
      )
      self.assertIsNone(res.subunit)
      self.assertEqual(captured_params[0], {})
      self.assertEqual(
          (captured_headers[0] or {}).get("X-Goog-Api-Key"),
          "AIzaSySecretKey12345",
      )

  async def test_api_key_redacted_and_transient_error_not_cached(self) -> None:
    attempt = 0

    async def flaky_post(
        url: str,
        params: dict[str, Any],
        payload: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del url, params, payload, timeout, headers
      nonlocal attempt
      attempt += 1
      if attempt == 1:
        raise RuntimeError(
            "Connection failed for https://example.com?key=AIzaSySecretKey12345"
        )
      return (
          200,
          json.dumps({
              "result": {
                  "address": {
                      "addressComponents": [
                          {
                              "componentType": "street_number",
                              "componentName": {"text": "10"},
                          },
                          {
                              "componentType": "route",
                              "componentName": {"text": "Oak St"},
                          },
                      ],
                      "missingComponentTypes": [],
                  },
                  "metadata": {"business": False},
              }
          }).encode("utf-8"),
      )

    with patch(
        "google.adk.integrations.street_view._services._async_http_post_json",
        side_effect=flaky_post,
    ):
      first = await classify_property_address_async(
          "10 Oak St", config=self.config
      )
      self.assertIsNotNone(first.maps_api_error)
      self.assertNotIn("AIzaSySecretKey12345", first.maps_api_error or "")
      self.assertIn("[REDACTED]", first.maps_api_error or "")

      second = await classify_property_address_async(
          "10 Oak St", config=self.config
      )
      self.assertIsNone(second.maps_api_error)
      self.assertEqual(second.standardized_address, "10 OAK ST")

  def test_address_validation_regex_fallback_preserves_city_state(self) -> None:
    unconfigured = GoogleMapsConfig(api_key=None)
    with patch.dict(os.environ, {}, clear=True):
      multi = classify_property_address(
          "742 Evergreen Terrace Apt 2B, Springfield, IL 62704",
          config=unconfigured,
      )
      self.assertEqual(multi.address_type, AddressType.MULTI_UNIT)
      self.assertEqual(
          multi.base_address,
          "742 EVERGREEN TERRACE, SPRINGFIELD, IL 62704",
      )
      self.assertEqual(
          multi.standardized_address,
          "742 EVERGREEN TERRACE, APT 2B, SPRINGFIELD, IL 62704",
      )
      self.assertEqual(multi.subunit, "APT 2B")
      self.assertEqual(multi.maps_api_error, "MAPS_API_KEY unconfigured")

      single = classify_property_address(
          "742 Evergreen Terrace, Springfield, IL 62704",
          config=unconfigured,
      )
      self.assertEqual(single.address_type, AddressType.SINGLE_FAMILY)
      self.assertIsNone(single.subunit)

  async def test_reverse_geocode_async_and_sync(self) -> None:
    fake_geo = {
        "status": "OK",
        "results": [{
            "formatted_address": (
                "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
            )
        }],
    }

    async def mock_get(
        url: str,
        params: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del url, params, timeout, headers
      return 200, json.dumps(fake_geo).encode("utf-8")

    with patch(
        "google.adk.integrations.street_view._services._async_http_get",
        side_effect=mock_get,
    ):
      addr = await reverse_geocode_coordinates_async(
          37.4220656, -122.0840897, config=self.config
      )
      self.assertEqual(
          addr, "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
      )
      cached_addr = reverse_geocode_coordinates(
          37.4220659, -122.0840899, config=self.config
      )
      self.assertEqual(cached_addr, addr)

  async def test_adk_tools_keep_session_state_json_serializable(self) -> None:
    fake_jpeg = b"\xff\xd8\xff\xe0STREET_VIEW_BYTES"

    async def mock_http_get(
        url: str,
        params: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del params, timeout, headers
      if "metadata" in url:
        return (
            200,
            json.dumps({
                "status": "OK",
                "pano_id": "pano_xyz",
                "date": "2026-01",
                "location": {"lat": 37.421, "lng": -122.084},
            }).encode("utf-8"),
        )
      if "geocode" in url:
        return (
            200,
            json.dumps({
                "status": "OK",
                "results": [{"formatted_address": "1600 Amphitheatre Pkwy"}],
            }).encode("utf-8"),
        )
      return 200, fake_jpeg

    ctx: Any = _DummyToolContext()
    with (
        patch(
            "google.adk.integrations.street_view._services._async_http_get",
            side_effect=mock_http_get,
        ),
        patch.dict(os.environ, {"MAPS_API_KEY": "test-key"}, clear=False),
    ):
      sv_out = await street_view_fallback_tool(37.422, -122.084, ctx)
      self.assertIn("Successfully captured Street View baseline", sv_out)

      invalid_out = await street_view_fallback_tool(999.0, -122.084, ctx)
      self.assertIn("Invalid GPS coordinates", invalid_out)

      prop_out = await property_classification_tool("123 Main St #4A", ctx)
      self.assertIn("MULTI_UNIT", prop_out)

      geo_out = await reverse_geocode_tool(37.422, -122.084, ctx)
      self.assertIn("1600 Amphitheatre Pkwy", geo_out)

      serialized_state = json.dumps(ctx.state)
      self.assertIsInstance(serialized_state, str)
      self.assertTrue(ctx.state["street_view_baseline_used"])
      self.assertEqual(
          ctx.state["street_view_artifact_name"],
          STREET_VIEW_ARTIFACT_FILENAME,
      )
      self.assertIn(STREET_VIEW_ARTIFACT_FILENAME, ctx.saved_artifacts)
      self.assertTrue(ctx.state["is_multi_unit"])
      self.assertEqual(
          ctx.state["reverse_geocoded_address"],
          "1600 Amphitheatre Pkwy",
      )

  async def test_street_view_toolset_filtering_and_signatures(self) -> None:
    toolset = StreetViewToolset(config=self.config)
    funcs = toolset.get_tool_functions()
    self.assertEqual(len(funcs), 3)

    for fn in funcs:
      sig = inspect.signature(fn)
      self.assertIn("tool_context", sig.parameters)
      self.assertNotIn("config", sig.parameters)

    tools = await toolset.get_tools()
    self.assertEqual(len(tools), 3)

    filtered_toolset = StreetViewToolset(
        config=self.config,
        tool_filter=["street_view_fallback_tool"],
    )
    filtered_tools = await filtered_toolset.get_tools()
    self.assertEqual(len(filtered_tools), 1)
    self.assertEqual(filtered_tools[0].name, "street_view_fallback_tool")

  async def test_audit_photo_with_fallback_clamps_confidence(self) -> None:
    fake_jpeg = b"\xff\xd8\xff\xe0TARGET_PHOTO"

    async def mock_http_get(
        url: str,
        params: dict[str, Any],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
      del params, timeout, headers
      if "metadata" in url:
        return (
            200,
            json.dumps({
                "status": "OK",
                "pano_id": "pano_audit",
                "date": "2026-02",
                "location": {"lat": 37.421, "lng": -122.084},
            }).encode("utf-8"),
        )
      return 200, fake_jpeg

    baseline_resp = MagicMock()
    baseline_resp.text = "Two-story brick home with white porch columns."
    verdict_resp = MagicMock()
    verdict_resp.parsed = None
    verdict_resp.text = (
        '```json\n{"verdict_label": "CONFIRMED_MATCH", "match": true, '
        '"confidence": 1.4, "reason": "Brick facade matches."}\n```'
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[baseline_resp, verdict_resp]
    )
    ctx: Any = _DummyToolContext()

    with patch(
        "google.adk.integrations.street_view._services._async_http_get",
        side_effect=mock_http_get,
    ):
      verdict = await audit_photo_with_fallback(
          target_photo_bytes=fake_jpeg,
          historical_photos=None,
          latitude=37.422,
          longitude=-122.084,
          tool_context=ctx,
          genai_client=mock_client,
          model="gemini-2.5-flash",
          config=self.config,
      )
      self.assertEqual(verdict["verdict_label"], "CONFIRMED_MATCH")
      self.assertTrue(verdict["match"])
      self.assertEqual(verdict["confidence"], 1.0)
      self.assertEqual(verdict["baseline_source"], "STREET_VIEW_FALLBACK")


if __name__ == "__main__":
  unittest.main()
