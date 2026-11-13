# Google Maps Platform Integration for ADK

`GoogleMapsToolset` gives ADK agents three Google Maps Platform tools:

| Tool | API | What it does |
| :--- | :--- | :--- |
| `get_street_view_image` | Street View Static API | Checks for outdoor coverage, points the camera at the location, and saves the image as a session artifact. |
| `validate_address` | Address Validation API | Standardizes an address, classifies it as `SINGLE_FAMILY` or `MULTI_UNIT`, and flags a missing unit number. |
| `reverse_geocode` | Geocoding API | Finds the street address at a latitude and longitude. |

## Setup

Enable the APIs on your Google Cloud project:

```shell
gcloud services enable \
  street-view-image-backend.googleapis.com \
  addressvalidation.googleapis.com \
  geocoding-backend.googleapis.com
```

Set the API key, or read it from Secret Manager:

```shell
export MAPS_API_KEY="your-maps-api-key"

# Optional: requires pip install 'google-adk[gcp]'
export MAPS_API_KEY_SECRET_RESOURCE="projects/my-project/secrets/maps-api-key/versions/latest"
```

## Usage

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

## Session state and artifacts

| Tool | Keys written on success |
| :--- | :--- |
| `get_street_view_image` | Artifact `street_view.jpg`; `street_view_artifact_name`, `street_view_available`, `street_view_url`, `street_view_pano_id`, `street_view_imagery_date`, `street_view_heading` |
| `validate_address` | `address_classification`, `is_multi_unit`, `is_missing_subunit`, `standardized_address` |
| `reverse_geocode` | `reverse_geocoded_address`, `reverse_geocoded_coordinates` |

When a request fails (missing key, API not enabled, HTTP error), the tool
returns the error and writes no state.
