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

"""Unit tests for the Google Maps Platform integration."""

import inspect
import json
import logging
import math
import os
import sys
from typing import Any
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from google.adk.integrations.google_maps import _tools
from google.adk.integrations.google_maps import GoogleMapsConfig
from google.adk.integrations.google_maps import GoogleMapsToolset
from google.adk.integrations.google_maps._models import AddressType
from google.adk.integrations.google_maps._services import _BoundedLRUCache
from google.adk.integrations.google_maps._services import calculate_compass_bearing
from google.adk.integrations.google_maps._services import classify_property_address_async
from google.adk.integrations.google_maps._services import clear_caches
from google.adk.integrations.google_maps._services import fetch_street_view_baseline
from google.adk.integrations.google_maps._services import MapsApiError
from google.adk.integrations.google_maps._services import reverse_geocode_coordinates_async
from google.adk.integrations.google_maps._services import validate_coordinates
import httpx

_SERVICES = "google.adk.integrations.google_maps._services"
_KEY = "AIzaSySecretKey12345"


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


def _json_get(response: dict[str, Any], status: int = 200):
  async def mock_get(
      url: str,
      params: dict[str, Any],
      timeout: float,
      headers: dict[str, str] | None = None,
  ) -> tuple[int, bytes]:
    del url, params, timeout, headers
    return status, json.dumps(response).encode("utf-8")

  return mock_get


def _json_post(response: dict[str, Any], status: int = 200):
  async def mock_post(
      url: str,
      params: dict[str, Any],
      payload: dict[str, Any],
      timeout: float,
      headers: dict[str, str] | None = None,
  ) -> tuple[int, bytes]:
    del url, params, payload, timeout, headers
    return status, json.dumps(response).encode("utf-8")

  return mock_post


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
  """Tests Street View, Address Validation, Geocoding, and the tools."""

  def setUp(self) -> None:
    super().setUp()
    clear_caches()
    self.config = GoogleMapsConfig(api_key=_KEY)
    self.unconfigured = GoogleMapsConfig(api_key=None)

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

    ctx: Any = _DummyToolContext()
    with patch(f"{_SERVICES}._async_http_get", side_effect=mock_http_get):
      result = await fetch_street_view_baseline(
          37.422, -122.084, config=self.config
      )
      out = await _tools.get_street_view_image(
          37.422, -122.084, ctx, config=self.config
      )
    self.assertFalse(result.available)
    self.assertIsNone(result.image_bytes)
    self.assertEqual(len(calls), 2)
    self.assertIn("streetview/metadata", calls[0][0])
    self.assertIn("not available", out)
    self.assertFalse(ctx.state["street_view_available"])

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

    with patch(f"{_SERVICES}._async_http_get", side_effect=mock_http_get):
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

  async def test_missing_api_key_is_reported_by_every_tool(self) -> None:
    ctx: Any = _DummyToolContext()
    mock_get = AsyncMock()
    mock_post = AsyncMock()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(f"{_SERVICES}._async_http_get", mock_get),
        patch(f"{_SERVICES}._async_http_post_json", mock_post),
    ):
      sv_out = await _tools.get_street_view_image(
          37.422, -122.084, ctx, config=self.unconfigured
      )
      addr_out = await _tools.validate_address(
          "742 Evergreen Terrace Apt 2B", ctx, config=self.unconfigured
      )
      geo_out = await _tools.reverse_geocode(
          37.422, -122.084, ctx, config=self.unconfigured
      )
    self.assertEqual(
        sv_out, "Street View lookup failed: MAPS_API_KEY unconfigured."
    )
    self.assertEqual(
        addr_out, "Address validation failed: MAPS_API_KEY unconfigured."
    )
    self.assertEqual(
        geo_out, "Reverse geocoding failed: MAPS_API_KEY unconfigured."
    )
    self.assertEqual(ctx.state, {})
    mock_get.assert_not_awaited()
    mock_post.assert_not_awaited()

  async def test_request_denied_is_reported_not_treated_as_no_result(
      self,
  ) -> None:
    denied = {
        "status": "REQUEST_DENIED",
        "error_message": "This API project is not authorized to use this API.",
    }
    ctx: Any = _DummyToolContext()
    with patch(f"{_SERVICES}._async_http_get", side_effect=_json_get(denied)):
      sv_out = await _tools.get_street_view_image(
          37.422, -122.084, ctx, config=self.config
      )
      geo_out = await _tools.reverse_geocode(
          37.422, -122.084, ctx, config=self.config
      )
    self.assertIn("Street View lookup failed", sv_out)
    self.assertIn("REQUEST_DENIED", sv_out)
    self.assertIn("not authorized", sv_out)
    self.assertIn("Reverse geocoding failed", geo_out)
    self.assertIn("REQUEST_DENIED", geo_out)
    self.assertEqual(ctx.state, {})

  async def test_http_error_is_reported_and_writes_no_state(self) -> None:
    forbidden = {
        "error": {
            "code": 403,
            "message": "Address Validation API has not been used.",
            "status": "PERMISSION_DENIED",
        }
    }
    ctx: Any = _DummyToolContext()
    with (
        patch(
            f"{_SERVICES}._async_http_post_json",
            side_effect=_json_post(forbidden, status=403),
        ),
        patch(
            f"{_SERVICES}._async_http_get",
            side_effect=_json_get({}, status=500),
        ),
    ):
      addr_out = await _tools.validate_address(
          "742 Evergreen Terrace", ctx, config=self.config
      )
      sv_out = await _tools.get_street_view_image(
          37.422, -122.084, ctx, config=self.config
      )
    self.assertEqual(
        addr_out,
        "Address validation failed: Address Validation returned HTTP 403:"
        " Address Validation API has not been used..",
    )
    self.assertIn("Street View metadata returned HTTP 500", sv_out)
    self.assertNotIn("SINGLE_FAMILY", addr_out)
    self.assertEqual(ctx.state, {})

  async def test_reverse_geocode_zero_results_is_not_an_error(self) -> None:
    ctx: Any = _DummyToolContext()
    with patch(
        f"{_SERVICES}._async_http_get",
        side_effect=_json_get({"status": "ZERO_RESULTS", "results": []}),
    ):
      out = await _tools.reverse_geocode(0.0, -150.0, ctx, config=self.config)
    self.assertIn("Could not resolve a street address", out)
    self.assertEqual(ctx.state, {})

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

    with patch(f"{_SERVICES}._async_http_post_json", side_effect=mock_post):
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
    self.assertEqual((captured_headers[0] or {}).get("X-Goog-Api-Key"), _KEY)

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
        raise httpx.ConnectError(
            f"Connection failed for https://example.com?key={_KEY}"
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

    with patch(f"{_SERVICES}._async_http_post_json", side_effect=flaky_post):
      with self.assertRaises(MapsApiError) as cm:
        await classify_property_address_async("10 Oak St", config=self.config)
      self.assertNotIn(_KEY, str(cm.exception))
      self.assertIn("[REDACTED]", str(cm.exception))

      second = await classify_property_address_async(
          "10 Oak St", config=self.config
      )
    self.assertEqual(second.standardized_address, "10 OAK ST")
    self.assertEqual(second.address_type, AddressType.SINGLE_FAMILY)

  async def test_address_validation_parse_error_propagates(self) -> None:
    with patch(
        f"{_SERVICES}._async_http_post_json",
        side_effect=_json_post({"result": []}),
    ):
      with self.assertRaises(AttributeError):
        await classify_property_address_async("10 Oak St", config=self.config)

  async def test_reverse_geocode_caches_rounded_coordinates(self) -> None:
    fake_geo = {
        "status": "OK",
        "results": [{
            "formatted_address": (
                "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
            )
        }],
    }
    mock_http = AsyncMock(side_effect=_json_get(fake_geo))
    with patch(f"{_SERVICES}._async_http_get", new=mock_http):
      addr = await reverse_geocode_coordinates_async(
          37.4220656, -122.0840897, config=self.config
      )
      cached_addr = await reverse_geocode_coordinates_async(
          37.4220659, -122.0840899, config=self.config
      )
    self.assertEqual(
        addr, "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
    )
    self.assertEqual(cached_addr, addr)
    self.assertEqual(mock_http.await_count, 1)

  async def test_api_key_not_in_httpx_request_log(self) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
      del request
      return httpx.Response(
          200,
          json={"status": "OK", "results": [{"formatted_address": "1 Main"}]},
      )

    with (
        patch(
            f"{_SERVICES}._mtls_utils.client_cert_httpx_kwargs",
            return_value={"transport": httpx.MockTransport(handler)},
        ),
        self.assertLogs("httpx", level=logging.INFO) as logs,
    ):
      addr = await reverse_geocode_coordinates_async(
          37.422, -122.084, config=self.config
      )
    self.assertEqual(addr, "1 Main")
    output = "\n".join(logs.output)
    self.assertIn("HTTP Request", output)
    self.assertIn("key=[REDACTED]", output)
    self.assertNotIn(_KEY, output)

  async def test_tools_keep_session_state_json_serializable(self) -> None:
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

    address_response = {
        "result": {
            "address": {
                "addressComponents": [
                    {
                        "componentType": "street_number",
                        "componentName": {"text": "123"},
                    },
                    {
                        "componentType": "route",
                        "componentName": {"text": "Main St"},
                    },
                    {
                        "componentType": "subpremise",
                        "componentName": {"text": "4A"},
                    },
                ],
            },
        }
    }

    ctx: Any = _DummyToolContext()
    tools = {
        tool.name: tool
        for tool in await GoogleMapsToolset(config=self.config).get_tools()
    }
    with (
        patch(f"{_SERVICES}._async_http_get", side_effect=mock_http_get),
        patch(
            f"{_SERVICES}._async_http_post_json",
            side_effect=_json_post(address_response),
        ),
    ):
      sv_out = await tools["get_street_view_image"].func(37.422, -122.084, ctx)
      invalid_out = await tools["get_street_view_image"].func(
          999.0, -122.084, ctx
      )
      addr_out = await tools["validate_address"].func("123 Main St #4A", ctx)
      geo_out = await tools["reverse_geocode"].func(37.422, -122.084, ctx)

    self.assertIn("Captured Street View imagery", sv_out)
    self.assertIn("Invalid coordinates", invalid_out)
    self.assertIn("MULTI_UNIT", addr_out)
    self.assertIn("1600 Amphitheatre Pkwy", geo_out)
    self.assertIsInstance(json.dumps(ctx.state), str)
    self.assertTrue(ctx.state["street_view_available"])
    self.assertEqual(
        ctx.state["street_view_artifact_name"],
        _tools.STREET_VIEW_ARTIFACT_FILENAME,
    )
    self.assertIn(_tools.STREET_VIEW_ARTIFACT_FILENAME, ctx.saved_artifacts)
    self.assertNotIn("street_view_image_b64", ctx.state)
    self.assertTrue(ctx.state["is_multi_unit"])
    self.assertEqual(
        ctx.state["reverse_geocoded_address"], "1600 Amphitheatre Pkwy"
    )

  async def test_toolset_names_filtering_and_signatures(self) -> None:
    toolset = GoogleMapsToolset(config=self.config)
    funcs = toolset._get_tool_functions()
    for fn in funcs:
      sig = inspect.signature(fn)
      self.assertIn("tool_context", sig.parameters)
      self.assertNotIn("config", sig.parameters)

    tools = await toolset.get_tools()
    self.assertEqual(
        [tool.name for tool in tools],
        ["get_street_view_image", "validate_address", "reverse_geocode"],
    )

    filtered_toolset = GoogleMapsToolset(
        config=self.config,
        tool_filter=["get_street_view_image"],
    )
    filtered_tools = await filtered_toolset.get_tools()
    self.assertEqual(
        [tool.name for tool in filtered_tools], ["get_street_view_image"]
    )


if __name__ == "__main__":
  unittest.main()
