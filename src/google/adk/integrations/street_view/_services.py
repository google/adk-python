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

from collections import OrderedDict
import json
import logging
import math
import re
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

STREET_VIEW_METADATA_URL = (
    "https://maps.googleapis.com/maps/api/streetview/metadata"
)
STREET_VIEW_METADATA_MTLS_URL = (
    "https://maps.mtls.googleapis.com/maps/api/streetview/metadata"
)
STREET_VIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
STREET_VIEW_IMAGE_MTLS_URL = (
    "https://maps.mtls.googleapis.com/maps/api/streetview"
)
ADDRESS_VALIDATION_URL = (
    "https://addressvalidation.googleapis.com/v1:validateAddress"
)
ADDRESS_VALIDATION_MTLS_URL = (
    "https://addressvalidation.mtls.googleapis.com/v1:validateAddress"
)
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GEOCODE_MTLS_URL = "https://maps.mtls.googleapis.com/maps/api/geocode/json"

_SUBUNIT_REGEX = re.compile(
    r"(?:\b(APT|APARTMENT|UNIT|STE|SUITE|BLDG|BUILDING|FL|FLOOR|RM|ROOM)"
    r"\b\.?\s*([A-Z0-9\-]+)?|#\s*([A-Z0-9\-]+))"
)
_QUERY_KEY_REDACT_RE = re.compile(r"([?&]key=)[^&\s\"']+", re.IGNORECASE)

K = TypeVar("K")
V = TypeVar("V")


class _BoundedLRUCache(Generic[K, V]):
  """In-memory bounded LRU cache using OrderedDict."""

  def __init__(self, maxsize: int = 1024) -> None:
    self._maxsize = maxsize
    self._store: OrderedDict[K, V] = OrderedDict()

  def get(self, key: K) -> tuple[bool, V | None]:
    if key not in self._store:
      return False, None
    self._store.move_to_end(key)
    return True, self._store[key]

  def put(self, key: K, value: V) -> None:
    if key in self._store:
      self._store.move_to_end(key)
    self._store[key] = value
    if len(self._store) > self._maxsize:
      self._store.popitem(last=False)

  def clear(self) -> None:
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


def _sync_mtls_client_and_url(url: str) -> tuple[dict[str, Any], str]:
  """Resolves httpx.Client mTLS kwargs and effective endpoint for sync calls."""
  mtls_kwargs = _mtls_utils.client_cert_httpx_kwargs(url=url)
  if "verify" not in mtls_kwargs:
    return {}, url
  return (
      {"verify": mtls_kwargs["verify"]},
      _mtls_utils.effective_googleapis_endpoint(url),
  )


def _sync_http_get(
    url: str,
    params: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
  _ensure_https_url(url)
  client_kwargs, target_url = _sync_mtls_client_and_url(url)
  with httpx.Client(timeout=timeout, **client_kwargs) as client:
    resp = client.get(target_url, params=params, headers=headers)
    return resp.status_code, resp.content


def _sync_http_post_json(
    url: str,
    params: dict[str, Any],
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
  _ensure_https_url(url)
  client_kwargs, target_url = _sync_mtls_client_and_url(url)
  with httpx.Client(timeout=timeout, **client_kwargs) as client:
    resp = client.post(target_url, params=params, json=payload, headers=headers)
    return resp.status_code, resp.content


async def _async_http_get(
    url: str,
    params: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
  _ensure_https_url(url)
  mtls_kwargs = _mtls_utils.client_cert_httpx_kwargs(url=url)
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
  mtls_kwargs = _mtls_utils.client_cert_httpx_kwargs(url=url)
  async with httpx.AsyncClient(timeout=timeout, **mtls_kwargs) as client:
    resp = await client.post(url, params=params, json=payload, headers=headers)
    return resp.status_code, resp.content


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
  """Fetches outdoor Street View imagery using the 2-step Metadata pattern."""
  if not validate_coordinates(latitude, longitude):
    logger.warning(
        "Invalid GPS coordinates (%s, %s); skipping Street View query.",
        latitude,
        longitude,
    )
    return StreetViewResult(available=False, coordinates=(latitude, longitude))

  cfg = config or default_config
  api_key = cfg.get_api_key()
  if not api_key:
    logger.info("Maps API key not available; skipping Street View baseline.")
    return StreetViewResult(available=False, coordinates=(latitude, longitude))

  req_timeout = timeout if timeout is not None else cfg.timeout
  req_pitch = pitch if pitch is not None else cfg.default_pitch
  req_fov = fov if fov is not None else cfg.default_fov
  req_size = size if size is not None else cfg.default_size

  meta_params = {
      "location": f"{latitude},{longitude}",
      "source": "outdoor",
      "key": api_key,
  }
  pano_id: str | None = None
  imagery_date: str | None = None
  camera_coords: tuple[float, float] | None = None
  computed_heading: float | None = heading

  try:
    meta_status, meta_body = await _async_http_get(
        STREET_VIEW_METADATA_URL, meta_params, req_timeout
    )
    if meta_status != 200:
      return StreetViewResult(
          available=False, coordinates=(latitude, longitude)
      )

    meta_json = json.loads(meta_body.decode("utf-8"))
    if meta_json.get("status") != "OK":
      logger.info(
          "No Street View imagery at (%s, %s): %s",
          latitude,
          longitude,
          meta_json.get("status"),
      )
      return StreetViewResult(
          available=False, coordinates=(latitude, longitude)
      )

    pano_id = meta_json.get("pano_id")
    imagery_date = meta_json.get("date")
    loc = meta_json.get("location")
    if isinstance(loc, dict) and "lat" in loc and "lng" in loc:
      cam_lat = float(loc["lat"])
      cam_lng = float(loc["lng"])
      camera_coords = (cam_lat, cam_lng)
      if computed_heading is None:
        computed_heading = calculate_compass_bearing(
            cam_lat, cam_lng, latitude, longitude
        )
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning(
        "Street View metadata check failed: %s",
        _redact_secret(str(exc), api_key),
    )
    return StreetViewResult(available=False, coordinates=(latitude, longitude))

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

  try:
    img_status, img_bytes = await _async_http_get(
        STREET_VIEW_IMAGE_URL, sv_params, req_timeout
    )
    if img_status == 200 and img_bytes:
      maps_url = (
          "https://www.google.com/maps/@?api=1&map_action=pano"
          f"&viewpoint={latitude},{longitude}"
      )
      if pano_id:
        maps_url += f"&pano={pano_id}"
      if computed_heading is not None:
        maps_url += (
            f"&heading={computed_heading}&pitch={req_pitch}&fov={req_fov}"
        )

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
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning(
        "Street View image download failed: %s",
        _redact_secret(str(exc), api_key),
    )

  return StreetViewResult(
      available=False,
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


def _fallback_regex_classify(
    raw_address: str, maps_error: str | None
) -> PropertyClassification:
  """Deterministic regex fallback when Address Validation API is unreachable."""
  cleaned = " ".join(raw_address.strip().upper().split())
  match = _SUBUNIT_REGEX.search(cleaned)
  if match:
    prefix = cleaned[: match.start()].strip().rstrip(",")
    suffix = cleaned[match.end() :].strip().lstrip(",").strip()
    sub = match.group(0).strip()
    if sub.startswith("#"):
      num = sub.lstrip("#").strip()
      sub = f"APT {num}" if num else "APT"

    base_parts = [p for p in (prefix, suffix) if p]
    base = ", ".join(base_parts) if base_parts else cleaned

    std_parts = [p for p in (prefix, sub, suffix) if p]
    std_addr = ", ".join(std_parts)

    return PropertyClassification(
        standardized_address=std_addr,
        base_address=base,
        address_type=AddressType.MULTI_UNIT,
        subunit=sub,
        location_type="RESIDENTIAL",
        is_missing_subunit=False,
        maps_api_error=maps_error,
    )

  return PropertyClassification(
      standardized_address=cleaned,
      base_address=cleaned,
      address_type=AddressType.SINGLE_FAMILY,
      subunit=None,
      location_type="RESIDENTIAL",
      is_missing_subunit=False,
      maps_api_error=maps_error,
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
  """Asynchronously standardizes an address and classifies property type."""
  cfg = config or default_config
  api_key = cfg.get_api_key()
  normalized_input = " ".join(raw_address.strip().upper().split())

  if api_key:
    cache_key = (normalized_input, cfg.get_key_fingerprint())
    hit, cached = _address_cache.get(cache_key)
    if hit and cached is not None:
      return cached

    try:
      status, body = await _async_http_post_json(
          ADDRESS_VALIDATION_URL,
          params={},
          payload={"address": {"addressLines": [raw_address]}},
          timeout=cfg.timeout,
          headers={"X-Goog-Api-Key": api_key},
      )
      if status == 200:
        data = json.loads(body.decode("utf-8"))
        classification = _parse_address_validation_response(data)
        _address_cache.put(cache_key, classification)
        return classification
      maps_error: str | None = f"HTTP {status}"
    except Exception as exc:  # pylint: disable=broad-except
      maps_error = _redact_secret(str(exc), api_key)
  else:
    maps_error = "MAPS_API_KEY unconfigured"

  return _fallback_regex_classify(raw_address, maps_error)


def classify_property_address(
    raw_address: str,
    config: GoogleMapsConfig | None = None,
) -> PropertyClassification:
  """Synchronous version of ``classify_property_address_async``."""
  cfg = config or default_config
  api_key = cfg.get_api_key()
  normalized_input = " ".join(raw_address.strip().upper().split())

  if api_key:
    cache_key = (normalized_input, cfg.get_key_fingerprint())
    hit, cached = _address_cache.get(cache_key)
    if hit and cached is not None:
      return cached

    try:
      status, body = _sync_http_post_json(
          ADDRESS_VALIDATION_URL,
          params={},
          payload={"address": {"addressLines": [raw_address]}},
          timeout=cfg.timeout,
          headers={"X-Goog-Api-Key": api_key},
      )
      if status == 200:
        data = json.loads(body.decode("utf-8"))
        classification = _parse_address_validation_response(data)
        _address_cache.put(cache_key, classification)
        return classification
      maps_error: str | None = f"HTTP {status}"
    except Exception as exc:  # pylint: disable=broad-except
      maps_error = _redact_secret(str(exc), api_key)
  else:
    maps_error = "MAPS_API_KEY unconfigured"

  return _fallback_regex_classify(raw_address, maps_error)


async def reverse_geocode_coordinates_async(
    latitude: float,
    longitude: float,
    config: GoogleMapsConfig | None = None,
) -> str | None:
  """Asynchronously resolves GPS coordinates into a street address."""
  if not validate_coordinates(latitude, longitude):
    return None

  cfg = config or default_config
  api_key = cfg.get_api_key()
  if not api_key:
    return None

  rounded_lat = round(latitude, 5)
  rounded_lng = round(longitude, 5)
  cache_key = (rounded_lat, rounded_lng, cfg.get_key_fingerprint())
  hit, cached = _geocode_cache.get(cache_key)
  if hit and cached is not None:
    return cached

  try:
    status, body = await _async_http_get(
        GEOCODE_URL,
        params={"latlng": f"{rounded_lat},{rounded_lng}", "key": api_key},
        timeout=cfg.timeout,
    )
    if status == 200:
      data = json.loads(body.decode("utf-8"))
      if data.get("status") == "OK" and data.get("results"):
        formatted = data["results"][0].get("formatted_address")
        if isinstance(formatted, str) and formatted:
          _geocode_cache.put(cache_key, formatted)
          return formatted
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning(
        "Reverse geocoding failed: %s",
        _redact_secret(str(exc), api_key),
    )
  return None


def reverse_geocode_coordinates(
    latitude: float,
    longitude: float,
    config: GoogleMapsConfig | None = None,
) -> str | None:
  """Synchronous version of ``reverse_geocode_coordinates_async``."""
  if not validate_coordinates(latitude, longitude):
    return None

  cfg = config or default_config
  api_key = cfg.get_api_key()
  if not api_key:
    return None

  rounded_lat = round(latitude, 5)
  rounded_lng = round(longitude, 5)
  cache_key = (rounded_lat, rounded_lng, cfg.get_key_fingerprint())
  hit, cached = _geocode_cache.get(cache_key)
  if hit and cached is not None:
    return cached

  try:
    status, body = _sync_http_get(
        GEOCODE_URL,
        params={"latlng": f"{rounded_lat},{rounded_lng}", "key": api_key},
        timeout=cfg.timeout,
    )
    if status == 200:
      data = json.loads(body.decode("utf-8"))
      if data.get("status") == "OK" and data.get("results"):
        formatted = data["results"][0].get("formatted_address")
        if isinstance(formatted, str) and formatted:
          _geocode_cache.put(cache_key, formatted)
          return formatted
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning(
        "Reverse geocoding failed: %s",
        _redact_secret(str(exc), api_key),
    )
  return None


def clear_caches() -> None:
  """Clears internal address validation and reverse geocoding caches."""
  _address_cache.clear()
  _geocode_cache.clear()
