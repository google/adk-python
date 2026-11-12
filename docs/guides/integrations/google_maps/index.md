# GoogleMapsToolset

`GoogleMapsToolset` connects ADK agents to Google Maps Platform: the Street View
Static API, the Address Validation API, and the Geocoding API.

## Get started

The toolset works with the base `google-adk` install. To read the API key from
Google Cloud Secret Manager, install the `gcp` extra:

```shell
pip install 'google-adk[gcp]'
```

Set `MAPS_API_KEY` (or pass `GoogleMapsConfig(api_key=...)`) and add the
toolset to an agent:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.google_maps import GoogleMapsConfig
from google.adk.integrations.google_maps import GoogleMapsToolset

agent = LlmAgent(
    name="maps_agent",
    instruction="Answer questions about addresses and locations.",
    tools=[GoogleMapsToolset(config=GoogleMapsConfig(default_fov=120))],
)
```

Use `tool_filter` to expose only some tools, for example
`GoogleMapsToolset(tool_filter=["validate_address"])`.

## Tools

### get_street_view_image

Queries the Street View metadata endpoint with `source="outdoor"` before it
requests an image, so locations without coverage cost no image quota. When
coverage exists, it points the camera from the panorama's position toward the
target coordinates. It saves the JPEG as the artifact `street_view.jpg` (or as
base64 in `state["street_view_image_b64"]` when no artifact service is set) and
writes `street_view_url`, `street_view_pano_id`, `street_view_imagery_date`,
and `street_view_heading` to state.

### validate_address

Sends the address to the Address Validation API with the `X-Goog-Api-Key`
header. It returns the standardized address, classifies it as `SINGLE_FAMILY`
or `MULTI_UNIT`, and sets `is_missing_subunit` when the API reports a missing
`subpremise` or `room`.

### reverse_geocode

Rounds the coordinates to 5 decimal places (about 1.1 m) and returns the first
Geocoding result. Successful lookups are cached in memory, partitioned by a
SHA-256 fingerprint of the API key.

## Errors

A missing key, an API that is not enabled, or a failed request comes back as
an error message from the tool, and the tool writes no state. "No coverage" and
"no address found" are normal results, not errors.

The Street View and Geocoding APIs accept the key only as a `key=` query
parameter. The integration redacts it from the `httpx` request log.

## Sample

[`google_maps_photo_audit`](../../../../contributing/samples/integrations/google_maps_photo_audit/)
uses the toolset and Gemini to check whether a photo was taken at a given
address.

## Limitations

- **Outdoor coverage only**: private roads and gated communities often have no
  outdoor Street View coverage.
- **Imagery age**: panoramas show conditions on the capture date in
  `state["street_view_imagery_date"]` (`YYYY-MM`), not today.
