# StreetViewToolset

`StreetViewToolset` connects ADK agents to Google Maps Platform APIs—Street View Static API, Address Validation API, and Reverse Geocoding API—for multimodal spatial verification, cold-start visual baselines, and property boundary classification.

## Introduction

Agents that verify physical locations (such as disputed deliveries, insurance claims, merchant storefront checks, or field-service inspections) often encounter two gaps:

1. **Cold-start visual baselines**: Newly serviced addresses have no historical ground-truth photos to compare against a submitted photo.
2. **Property boundary ambiguity**: A single-family home has a tight drop-off boundary around its porch or garage, whereas a multi-unit apartment complex has distributed drop-off locations across lobbies, mailrooms, and parcel lockers.

`StreetViewToolset` bundles three tools (`street_view_fallback_tool`, `property_classification_tool`, and `reverse_geocode_tool`) backed by `GoogleMapsConfig` and structured Pydantic models (`StreetViewResult`, `PropertyClassification`, and `VisualAuditVerdict`).

## Get started

Core HTTP and validation features work out of the box with `google-adk`. If you resolve API keys from Google Cloud Secret Manager, install the `gcp` extra:

```shell
pip install 'google-adk[gcp]'
```

Configure `MAPS_API_KEY` (or pass `GoogleMapsConfig`) and attach `StreetViewToolset` to an `LlmAgent`:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.street_view import GoogleMapsConfig
from google.adk.integrations.street_view import StreetViewToolset
from google.adk.integrations.street_view import default_config

agent = LlmAgent(
    name="spatial_auditor",
    instruction=(
        "Classify the recipient address, verify GPS telemetry with reverse "
        "geocoding, and capture a Street View baseline when prior imagery is "
        "unavailable."
    ),
    tools=[
        StreetViewToolset(
            config=GoogleMapsConfig(
                default_pitch=default_config.default_pitch,
                default_fov=120,
            )
        )
    ],
)
```

You can also register individual standalone tool functions directly on an agent:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.street_view import property_classification_tool
from google.adk.integrations.street_view import reverse_geocode_tool
from google.adk.integrations.street_view import street_view_fallback_tool

agent = LlmAgent(
    name="lightweight_auditor",
    tools=[
        street_view_fallback_tool,
        property_classification_tool,
        reverse_geocode_tool,
    ],
)
```

## How it works

### Two-step Street View pre-flight and compass bearing

`fetch_street_view_baseline` queries the Street View Metadata endpoint (`source="outdoor"`) before requesting image tiles:

1. Coordinates are checked with `validate_coordinates` to ensure finite WGS84 bounds.
2. If no outdoor panorama exists (`ZERO_RESULTS`), the call returns `StreetViewResult(available=False, ...)` immediately without billing Static API image quota or downloading gray placeholder tiles.
3. When outdoor imagery exists, `calculate_compass_bearing` computes the forward azimuth (0–360°) from the Street View camera vehicle's road coordinates toward the target property coordinates so the panorama faces the building rather than down the street.
4. `street_view_fallback_tool` saves the JPEG bytes as an ADK artifact named `STREET_VIEW_ARTIFACT_FILENAME` (`"street_view_baseline.jpg"`) and writes JSON-safe metadata (`street_view_url`, `street_view_pano_id`, `street_view_imagery_date`, `street_view_heading`) into `tool_context.state`.

### Address validation and regex fallback

`classify_property_address_async` (and synchronous `classify_property_address`) sends the raw address to the Address Validation API (`validateAddress`) using the `X-Goog-Api-Key` header:

- Extracts normalized street, locality, state, postal code, and subpremise components into `PropertyClassification` with `AddressType.SINGLE_FAMILY` or `AddressType.MULTI_UNIT`.
- Checks `missingComponentTypes` for `"subpremise"` or `"room"` to flag multi-unit buildings where the customer omitted an apartment or suite number (`is_missing_subunit=True`).
- Falls back to deterministic regex sub-unit parsing if `MAPS_API_KEY` is unconfigured or the API returns a transient error (without caching failed lookups).

### Reverse geocoding and caching

`reverse_geocode_coordinates_async` (and synchronous `reverse_geocode_coordinates`) resolves GPS telemetry rounded to 5 decimal places (~1.1 m precision) into a formatted street address. Successful lookups are stored in a bounded LRU cache partitioned by a non-reversible SHA-256 key fingerprint; call `clear_caches` in tests or credential rotation hooks to reset cached entries.

### Multimodal baseline comparison

For custom workflows or tool pipelines, `audit_photo_with_fallback` compares a target photo against historical photos (or automatically falls back to Street View when `historical_photos` is empty) using `BASELINE_SYNTHESIS_PROMPT` and `AUDIT_VERDICT_PROMPT`, returning a structured `VisualAuditVerdict` dictionary classified by `VerdictLabel`:

```python
from google import genai
from google.adk.integrations.street_view import AUDIT_VERDICT_PROMPT
from google.adk.integrations.street_view import BASELINE_SYNTHESIS_PROMPT
from google.adk.integrations.street_view import STREET_VIEW_ARTIFACT_FILENAME
from google.adk.integrations.street_view import audit_photo_with_fallback
from google.adk.integrations.street_view import calculate_compass_bearing
from google.adk.integrations.street_view import classify_property_address
from google.adk.integrations.street_view import classify_property_address_async
from google.adk.integrations.street_view import clear_caches
from google.adk.integrations.street_view import fetch_street_view_baseline
from google.adk.integrations.street_view import reverse_geocode_coordinates
from google.adk.integrations.street_view import reverse_geocode_coordinates_async
from google.adk.integrations.street_view import validate_coordinates


async def run_visual_audit(target_jpeg: bytes, tool_context):
    client = genai.Client()
    verdict = await audit_photo_with_fallback(
        target_photo_bytes=target_jpeg,
        historical_photos=None,
        latitude=37.4220656,
        longitude=-122.0840897,
        tool_context=tool_context,
        genai_client=client,
    )
    return verdict
```

## Public API reference

| Export | Description |
| :--- | :--- |
| `StreetViewToolset` | `BaseToolset` implementation exposing configurable Maps & Street View `FunctionTool` instances with `tool_filter` and `tool_name_prefix` support. |
| `GoogleMapsConfig` | Credential and request configuration resolving keys from `api_key`, `MAPS_API_KEY`, or Secret Manager (`MAPS_API_KEY_SECRET_RESOURCE`). |
| `default_config` | Module-level default `GoogleMapsConfig` instance used by standalone tools. |
| `street_view_fallback_tool` | Async ADK tool function that captures outdoor Street View imagery into session artifacts and state. |
| `property_classification_tool` | Async ADK tool function that standardizes addresses and classifies single-family vs. multi-unit boundaries. |
| `reverse_geocode_tool` | Async ADK tool function that resolves GPS coordinates into a street address in session state. |
| `audit_photo_with_fallback` | Two-stage Gemini Vision helper that synthesizes a visual baseline and scores a target photo. |
| `fetch_street_view_baseline` | Low-level async service executing the Street View Metadata pre-flight check and image fetch. |
| `classify_property_address_async` | Async Address Validation service with deterministic local regex fallback. |
| `classify_property_address` | Synchronous counterpart to `classify_property_address_async`. |
| `reverse_geocode_coordinates_async` | Async Reverse Geocoding service with bounded LRU caching. |
| `reverse_geocode_coordinates` | Synchronous counterpart to `reverse_geocode_coordinates_async`. |
| `calculate_compass_bearing` | Computes forward compass heading (0–360°) from camera coordinates to target coordinates. |
| `validate_coordinates` | Validates finite WGS84 latitude `[-90, 90]` and longitude `[-180, 180]` bounds. |
| `clear_caches` | Clears the in-memory bounded LRU caches for address validation and reverse geocoding. |
| `AddressType` | `StrEnum` with `SINGLE_FAMILY` and `MULTI_UNIT` values. |
| `PropertyClassification` | Pydantic model representing normalized address components and sub-unit flags. |
| `StreetViewResult` | Pydantic model holding panorama bytes, deep-link URL, `pano_id`, `imagery_date`, and `heading`. |
| `VerdictLabel` | `Literal` union of `CONFIRMED_MATCH`, `CONFIRMED_MISMATCH`, `PLAUSIBLE_ALTERNATE_LOCATION`, and `INSUFFICIENT_VISUAL_CONTEXT`. |
| `VisualAuditVerdict` | Pydantic model returned by multimodal baseline comparison. |
| `STREET_VIEW_ARTIFACT_FILENAME` | Default ADK artifact filename (`"street_view_baseline.jpg"`). |
| `BASELINE_SYNTHESIS_PROMPT` | Reference prompt template for extracting stable architectural features from baseline imagery. |
| `AUDIT_VERDICT_PROMPT` | Reference prompt template for scoring target imagery against property boundaries. |

## Limitations

- **Outdoor coverage only**: `fetch_street_view_baseline` filters with `source="outdoor"` to avoid indoor photospheres, so gated communities or private roads without public Street View coverage return `available=False`.
- **Imagery recency**: Street View panoramas reflect the capture date reported in `StreetViewResult.imagery_date` (`YYYY-MM`) rather than real-time conditions; visual comparison prompts therefore focus on stable architectural features rather than transient objects or seasonal foliage.
