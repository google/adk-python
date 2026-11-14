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

"""Core Google Maps Platform service functions for ADK multimodal agents."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable
import json
import logging
import math
import re
import threading
from typing import Any
from typing import Generic
from typing import TypeVar
import urllib.parse

import httpx

from ...utils import _mtls_utils
from ._config import default_config
from ._config import GoogleMapsConfig
from ._models import AddressType
from ._models import PropertyClassification
from ._models import StreetViewResult

logger = logging.getLogger("google_adk." + __name__)

# With a client certificate, _mtls_utils sends these to their
# *.mtls.googleapis.com hosts.
STREET_VIEW_METADATA_URL = (
    "https://maps.googleapis.com/maps/api/streetview/metadata"
)
STREET_VIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
ADDRESS_VALIDATION_URL = (
    "https://addressvalidation.googleapis.com/v1:validateAddress"
)
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_QUERY_KEY_REDACT_RE = re.compile(r"([?&]key=)[^&\s\"']+", re.IGNORECASE)
# Statuses that mean "no match", as opposed to a failed request.
_NO_RESULT_STATUSES = frozenset({"ZERO_RESULTS", "NOT_FOUND"})


class _RedactApiKeyFilter(logging.Filter):
  """Redacts ``key=`` query values from httpx request logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    try:
      message = record.getMessage()
    except Exception:  # pylint: disable=broad-except
      return True
    redacted = _QUERY_KEY_REDACT_RE.sub(r"\1[REDACTED]", message)
    if redacted != message:
      record.msg = redacted
      record.args = None
    return True


# Street View and Geocoding only accept the key as a query parameter, and httpx
# logs the full request URL at INFO.
_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(f, _RedactApiKeyFilter) for f in _httpx_logger.filters):
  _httpx_logger.addFilter(_RedactApiKeyFilter())


class MapsApiError(Exception):
  """A Google Maps Platform request failed or could not be sent."""


K = TypeVar("K")
V = TypeVar("V")


class _BoundedLRUCache(Generic[K, V]):
  """In-memory bounded LRU cache using OrderedDict."""

  def __init__(self, maxsize: int = 1024) -> None:
    self._maxsize = maxsize
    self._store: OrderedDict[K, V] = OrderedDict()
    self._lock = threading.Lock()

  def get(self, key: K) -> tuple[bool, V | None]:
    with self._lock:
      if key not in self._store:
        return False, None
      self._store.move_to_end(key)
      return True, self._store[key]

  def put(self, key: K, value: V) -> None:
    with self._lock:
      if key in self._store:
        self._store.move_to_end(key)
      self._store[key] = value
      if len(self._store) > self._maxsize:
        self._store.popitem(last=False)

  def clear(self) -> None:
    with self._lock:
      self._store.clear()


def _redact_secret(text: str, api_key: str | None) -> str:
  """Removes API keys from exception or diagnostic strings."""
  sanitized = _QUERY_KEY_REDACT_RE.sub(r"\1[REDACTED]", text)
  if api_key:
    sanitized = sanitized.replace(api_key, "[REDACTED]")
  return sanitized


def validate_coordinates(latitude: float, longitude: float) -> bool:
  """Returns True if latitude and longitude are finite within WGS84 bounds."""
  if not (math.isfinite(latitude) and math.isfinite(longitude)):
    return False
  return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def calculate_compass_bearing(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> float | None:
  """Calculates forward compass bearing (0-360 deg) from start to end GPS.

  Used to orient the Street View camera from its road capture position
  directly toward the target property coordinates. Returns ``None`` if both
  points are identical or invalid.
  """
  if not (
      validate_coordinates(start_lat, start_lng)
      and validate_coordinates(end_lat, end_lng)
  ):
    return None
  if abs(start_lat - end_lat) < 1e-6 and abs(start_lng - end_lng) < 1e-6:
    return None

  lat1 = math.radians(start_lat)
  lat2 = math.radians(end_lat)
  diff_lng = math.radians(end_lng - start_lng)

  x = math.sin(diff_lng) * math.cos(lat2)
  y = math.cos(lat1) * math.sin(lat2) - (
      math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng)
  )
  initial_bearing = math.degrees(math.atan2(x, y))
  return round((initial_bearing + 360.0) % 360.0, 1)


def _ensure_https_url(url: str) -> None:
  parsed = urllib.parse.urlparse(url)
  if parsed.scheme != "https" or not parsed.netloc:
    raise ValueError(f"Only HTTPS Google Maps endpoints are allowed: {url}")


async def _async_http_get(
    url: str,
    params: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
  _ensure_https_url(url)
  mtls_kwargs = await asyncio.to_thread(
      _mtls_utils.client_cert_httpx_kwargs, url=url
  )
  async with httpx.AsyncClient(timeout=timeout, **mtls_kwargs) as client:
    resp = await client.get(url, params=params, headers=headers)
    return resp.status_code, resp.content


async def _async_http_post_json(
    url: str,
    params: dict[str, Any],
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
  _ensure_https_url(url)
  mtls_kwargs = await asyncio.to_thread(
      _mtls_utils.client_cert_httpx_kwargs, url=url
  )
  async with httpx.AsyncClient(timeout=timeout, **mtls_kwargs) as client:
    resp = await client.post(url, params=params, json=payload, headers=headers)
    return resp.status_code, resp.content


async def _require_api_key(cfg: GoogleMapsConfig) -> str:
  api_key = await cfg.get_api_key_async()
  if not api_key:
    raise MapsApiError("MAPS_API_KEY unconfigured")
  return api_key


def _error_detail(body: bytes) -> str:
  """Returns the error message from a Maps API JSON body, if any."""
  try:
    data = json.loads(body.decode("utf-8"))
  except ValueError:
    return ""
  if not isinstance(data, dict):
    return ""
  err = data.get("error")
  if isinstance(err, dict):
    return str(err.get("message") or err.get("status") or "")
  return str(data.get("error_message") or "")


async def _send(
    api: str, api_key: str, request: Awaitable[tuple[int, bytes]]
) -> bytes:
  """Awaits a request and returns its body, raising MapsApiError on failure."""
  try:
    status, body = await request
  except httpx.HTTPError as exc:
    raise MapsApiError(
        f"{api} request failed: {_redact_secret(str(exc), api_key)}"
    ) from None
  if status != 200:
    detail = _redact_secret(_error_detail(body), api_key)
    raise MapsApiError(
        f"{api} returned HTTP {status}" + (f": {detail}" if detail else "")
    )
  return body


def _parse_json(api: str, body: bytes) -> dict[str, Any]:
  try:
    data = json.loads(body.decode("utf-8"))
  except ValueError:
    raise MapsApiError(f"{api} returned invalid JSON") from None
  if not isinstance(data, dict):
    raise MapsApiError(f"{api} returned invalid JSON")
  return data


def _status_error(api: str, data: dict[str, Any], api_key: str) -> MapsApiError:
  message = f"{api} returned {data.get('status')}"
  detail = data.get("error_message")
  if detail:
    message += f": {_redact_secret(str(detail), api_key)}"
  return MapsApiError(message)


async def fetch_street_view_baseline(
    latitude: float,
    longitude: float,
    config: GoogleMapsConfig | None = None,
    timeout: float | None = None,
    pitch: int | None = None,
    fov: int | None = None,
    size: str | None = None,
    heading: float | None = None,
) -> StreetViewResult:
  """Fetches outdoor Street View imagery using the 2-step Metadata pattern.

  Returns ``available=False`` when there is no outdoor coverage.

  Raises:
    MapsApiError: If the key is missing or a request fails.
  """
  if not validate_coordinates(latitude, longitude):
    logger.warning(
        "Invalid GPS coordinates (%s, %s); skipping Street View query.",
        latitude,
        longitude,
    )
    return StreetViewResult(available=False, coordinates=(latitude, longitude))

  cfg = config or default_config
  api_key = await _require_api_key(cfg)

  req_timeout = timeout if timeout is not None else cfg.timeout
  req_pitch = pitch if pitch is not None else cfg.default_pitch
  req_fov = fov if fov is not None else cfg.default_fov
  req_size = size if size is not None else cfg.default_size

  meta_params = {
      "location": f"{latitude},{longitude}",
      "source": "outdoor",
      "key": api_key,
  }
  meta_body = await _send(
      "Street View metadata",
      api_key,
      _async_http_get(STREET_VIEW_METADATA_URL, meta_params, req_timeout),
  )
  meta_json = _parse_json("Street View metadata", meta_body)
  meta_status = meta_json.get("status")
  if meta_status in _NO_RESULT_STATUSES:
    logger.info(
        "No Street View imagery at (%s, %s): %s",
        latitude,
        longitude,
        meta_status,
    )
    return StreetViewResult(available=False, coordinates=(latitude, longitude))
  if meta_status != "OK":
    raise _status_error("Street View metadata", meta_json, api_key)

  pano_id: str | None = meta_json.get("pano_id")
  imagery_date: str | None = meta_json.get("date")
  camera_coords: tuple[float, float] | None = None
  computed_heading: float | None = heading
  loc = meta_json.get("location")
  if isinstance(loc, dict) and "lat" in loc and "lng" in loc:
    cam_lat = float(loc["lat"])
    cam_lng = float(loc["lng"])
    camera_coords = (cam_lat, cam_lng)
    if computed_heading is None:
      computed_heading = calculate_compass_bearing(
          cam_lat, cam_lng, latitude, longitude
      )

  sv_params: dict[str, Any] = {
      "size": req_size,
      "pitch": req_pitch,
      "fov": req_fov,
      "source": "outdoor",
      "key": api_key,
  }
  if pano_id and computed_heading is not None:
    sv_params["pano"] = pano_id
    sv_params["heading"] = computed_heading
  else:
    sv_params["location"] = f"{latitude},{longitude}"
    if pano_id:
      sv_params["pano"] = pano_id
    if computed_heading is not None:
      sv_params["heading"] = computed_heading

  img_bytes = await _send(
      "Street View image",
      api_key,
      _async_http_get(STREET_VIEW_IMAGE_URL, sv_params, req_timeout),
  )
  if not img_bytes:
    raise MapsApiError("Street View image returned no data")

  maps_url = (
      "https://www.google.com/maps/@?api=1&map_action=pano"
      f"&viewpoint={latitude},{longitude}"
  )
  if pano_id:
    maps_url += f"&pano={pano_id}"
  if computed_heading is not None:
    maps_url += f"&heading={computed_heading}&pitch={req_pitch}&fov={req_fov}"

  return StreetViewResult(
      available=True,
      image_bytes=img_bytes,
      maps_url=maps_url,
      coordinates=(latitude, longitude),
      pano_id=pano_id,
      imagery_date=imagery_date,
      camera_coordinates=camera_coords,
      heading=computed_heading,
  )


def _parse_address_validation_response(
    data: dict[str, Any],
) -> PropertyClassification:
  """Parses a Google Address Validation API v1 JSON response."""
  result = data.get("result", {})
  address_data = result.get("address", {})
  components = address_data.get("addressComponents", [])
  missing = address_data.get("missingComponentTypes", [])
  is_business = bool(result.get("metadata", {}).get("business", False))
  location_type = "BUSINESS" if is_business else "RESIDENTIAL"

  subunit: str | None = None
  street_number = ""
  route = ""
  locality = ""
  admin_area = ""
  postal_code = ""

  for comp in components:
    ctype = comp.get("componentType")
    cname = comp.get("componentName", {}).get("text", "").strip().upper()
    if not cname:
      continue
    if ctype in ("subpremise", "room"):
      if any(
          cname.startswith(prefix)
          for prefix in ("APT", "UNIT", "STE", "SUITE", "RM", "ROOM", "#")
      ):
        subunit = cname
      else:
        subunit = f"APT {cname}"
    elif ctype == "street_number":
      street_number = cname
    elif ctype == "route":
      route = cname
    elif ctype == "locality":
      locality = cname
    elif ctype == "administrative_area_level_1":
      admin_area = cname
    elif ctype == "postal_code":
      postal_code = cname

  street_line = f"{street_number} {route}".strip().upper()
  if not street_line:
    formatted = address_data.get("formattedAddress", "")
    street_line = (
        formatted.split(",")[0].strip().upper() if formatted else "UNKNOWN"
    )

  state_zip = f"{admin_area} {postal_code}".strip()
  city_state_zip = ", ".join(p for p in (locality, state_zip) if p)
  base = f"{street_line}, {city_state_zip}" if city_state_zip else street_line

  is_missing_sub = "subpremise" in missing or "room" in missing
  if subunit or is_missing_sub:
    if subunit:
      std_parts = [street_line, subunit]
      if city_state_zip:
        std_parts.append(city_state_zip)
      std_addr = ", ".join(std_parts)
    else:
      std_addr = base
    return PropertyClassification(
        standardized_address=std_addr,
        base_address=base,
        address_type=AddressType.MULTI_UNIT,
        subunit=subunit,
        location_type=location_type,
        is_missing_subunit=is_missing_sub,
    )

  return PropertyClassification(
      standardized_address=base,
      base_address=base,
      address_type=AddressType.SINGLE_FAMILY,
      subunit=None,
      location_type=location_type,
      is_missing_subunit=False,
  )


_address_cache: _BoundedLRUCache[tuple[str, str], PropertyClassification] = (
    _BoundedLRUCache(maxsize=1024)
)
_geocode_cache: _BoundedLRUCache[tuple[float, float, str], str] = (
    _BoundedLRUCache(maxsize=1024)
)


async def classify_property_address_async(
    raw_address: str,
    config: GoogleMapsConfig | None = None,
) -> PropertyClassification:
  """Standardizes an address and classifies the property type.

  Raises:
    MapsApiError: If the key is missing or the request fails.
  """
  cfg = config or default_config
  api_key = await _require_api_key(cfg)
  normalized_input = " ".join(raw_address.strip().upper().split())
  cache_key = (normalized_input, cfg.get_key_fingerprint())
  hit, cached = _address_cache.get(cache_key)
  if hit and cached is not None:
    return cached

  body = await _send(
      "Address Validation",
      api_key,
      _async_http_post_json(
          ADDRESS_VALIDATION_URL,
          params={},
          payload={"address": {"addressLines": [raw_address]}},
          timeout=cfg.timeout,
          headers={"X-Goog-Api-Key": api_key},
      ),
  )
  classification = _parse_address_validation_response(
      _parse_json("Address Validation", body)
  )
  _address_cache.put(cache_key, classification)
  return classification


async def reverse_geocode_coordinates_async(
    latitude: float,
    longitude: float,
    config: GoogleMapsConfig | None = None,
) -> str | None:
  """Resolves coordinates into a street address.

  Returns None when the coordinates are invalid or match no address.

  Raises:
    MapsApiError: If the key is missing or the request fails.
  """
  if not validate_coordinates(latitude, longitude):
    return None

  cfg = config or default_config
  api_key = await _require_api_key(cfg)

  rounded_lat = round(latitude, 5)
  rounded_lng = round(longitude, 5)
  cache_key = (rounded_lat, rounded_lng, cfg.get_key_fingerprint())
  hit, cached = _geocode_cache.get(cache_key)
  if hit and cached is not None:
    return cached

  body = await _send(
      "Geocoding",
      api_key,
      _async_http_get(
          GEOCODE_URL,
          params={"latlng": f"{rounded_lat},{rounded_lng}", "key": api_key},
          timeout=cfg.timeout,
      ),
  )
  data = _parse_json("Geocoding", body)
  status = data.get("status")
  if status in _NO_RESULT_STATUSES:
    return None
  if status != "OK":
    raise _status_error("Geocoding", data, api_key)
  results = data.get("results") or []
  formatted = results[0].get("formatted_address") if results else None
  if not isinstance(formatted, str) or not formatted:
    return None
  _geocode_cache.put(cache_key, formatted)
  return formatted


def clear_caches() -> None:
  """Clears internal address validation and reverse geocoding caches."""
  _address_cache.clear()
  _geocode_cache.clear()
