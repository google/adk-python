# Street View & Google Maps Platform Integration for ADK

Integrates **Google Maps Platform APIs** (Street View Static API, Address Validation API, and Geocoding API) with **Google GenAI / Gemini Vision** for multimodal spatial and physical verification workflows in ADK.

## Architecture Overview

In multimodal spatial and physical verification workflows (such as proof-of-delivery audits, field inspections, real estate assessments, and insurance claim reviews), agents often face two distinct challenges:

1. **Cold-Start Visual Disconnect ($N = 0$)**: When an agent needs to visually audit a target scene against historical baseline photos, newly serviced addresses or sparse historical data offer zero prior imagery.
2. **Ambiguous Property Boundaries**: Knowing whether an address is a **single-family home** (tight 50–150 ft drop-off boundary) or a **multi-unit apartment complex** (loose, distributed drop-off points across leasing offices, parcel lockers, or building clusters) determines the spatial decision boundary.

```text
                      ┌──────────────────────────────────────────────┐
                      │              Incoming Event                  │
                      │       (GPS Lat/Lng, Raw Address)             │
                      └──────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴──────────────────────┐
                       ▼                                            ▼
       ┌───────────────────────────────┐            ┌───────────────────────────────┐
       │   Address Validation API      │            │   Reverse Geocoding API       │
       │ (Standardize, classify unit,  │            │ (Resolve drop-off GPS into    │
       │  detect omitted apartment #)  │            │  physical street address)     │
       └───────────────┬───────────────┘            └───────────────┬───────────────┘
                       │                                            │
                       ▼                                            ▼
               [Property Class]                              [Street Context]
             (SINGLE_FAMILY vs.                            (Saved to ADK State)
                 MULTI_UNIT)                                        │
                       │                                            │
                       └─────────────────────┬──────────────────────┘
                                             │
                       ┌─────────────────────┴──────────────────────┐
                       │       Historical Imagery Available?        │
                       └──────┬──────────────────────────────┬──────┘
                       Yes    │                              │ No (Cold Start)
                              ▼                              ▼
                 ┌──────────────────────────┐   ┌───────────────────────────┐
                 │ GCS Historical Deliveries│   │ Street View Static API    │
                 │   (1..N Prior Photos)    │   │  • Pre-flight metadata chk│
                 └────────────┬─────────────┘   │  • Compute camera heading │
                              │                 │  • Fetch outdoor panorama │
                              │                 │  • Save ADK Artifact      │
                              │                 └─────────────┬─────────────┘
                              │                               │
                              └──────────────┬────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    Gemini Vision Baseline   │
                              │   (Generate visual features │
                              │    + score target photo)    │
                              └─────────────────────────────┘
```

## Enterprise Use Cases

| Vertical | Use Case | How `integrations.street_view` Solves It |
| :--- | :--- | :--- |
| **Logistics & Delivery** | Disputed Delivery ("Item Not Received") Audits | Classifies `SINGLE_FAMILY` vs. `MULTI_UNIT` boundaries, catches missing apartment numbers, reverse-geocodes driver GPS telemetry, and pulls Street View cold-start baselines ($N=0$). |
| **Property & Casualty Insurance** | Exterior Claim Triage & Underwriting | Captures pre-loss outdoor Street View imagery (including `imagery_date` metadata) to verify claimant-submitted property photos. |
| **FinTech & Marketplaces** | Merchant KYB & Storefront Verification | Validates `BUSINESS` vs. `RESIDENTIAL` address metadata and captures curb-view exterior imagery to confirm physical commercial signage. |
| **Field Services & Utilities** | Work-Order Completion Verification | Confirms technician inspection photos (EV chargers, solar installs, telecom drops) align with dispatched coordinates and surrounding streetscape. |
| **Real Estate & Lending** | Automated Desktop Appraisal Checks | Normalizes parcel/unit addresses and orients the Street View camera directly toward the target coordinates (`heading`) to cross-check listing imagery. |

## Prerequisites & Setup

Enable the required Google Maps Platform APIs on your Google Cloud project:

- **Street View Static API**
- **Address Validation API**
- **Geocoding API**

```shell
gcloud services enable \
  streetviewpublish.googleapis.com \
  static-maps-backend.googleapis.com \
  addressvalidation.googleapis.com \
  geocoding-backend.googleapis.com
```

Configure credentials via environment variable or Google Cloud Secret Manager:

```shell
export MAPS_API_KEY="your-maps-api-key"

# Optional: fetch from Secret Manager instead (requires pip install 'google-adk[gcp]')
export MAPS_API_KEY_SECRET_RESOURCE="projects/my-project/secrets/maps-api-key/versions/latest"
```

## Usage

### Registering `StreetViewToolset` on an ADK Agent

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.street_view import GoogleMapsConfig
from google.adk.integrations.street_view import StreetViewToolset

agent = LlmAgent(
    name="spatial_auditor",
    instruction="Verify delivery addresses, GPS telemetry, and outdoor imagery.",
    tools=[
        StreetViewToolset(
            config=GoogleMapsConfig(default_pitch=10, default_fov=120),
        )
    ],
)
```

### Session State & Artifacts Written by Tools

| Tool | Session State / Artifact Keys | Description |
| :--- | :--- | :--- |
| `street_view_fallback_tool` | Artifact `street_view_baseline.jpg`, `state["street_view_baseline_used"]`, `state["street_view_url"]`, `state["street_view_pano_id"]`, `state["street_view_imagery_date"]`, `state["street_view_heading"]` | Pre-flight checks `source="outdoor"`, computes compass bearing from the camera vehicle to target coordinates, and persists the panorama. |
| `property_classification_tool` | `state["address_classification"]`, `state["is_multi_unit"]`, `state["is_missing_subunit"]`, `state["standardized_address"]` | Standardizes the street address, classifies `SINGLE_FAMILY` vs. `MULTI_UNIT`, and detects omitted sub-units. |
| `reverse_geocode_tool` | `state["reverse_geocoded_address"]`, `state["dropoff_coordinates"]` | Resolves raw GPS latitude/longitude into a formatted street address. |
