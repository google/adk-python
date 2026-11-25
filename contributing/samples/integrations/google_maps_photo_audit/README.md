# Google Maps Photo Audit Sample

This agent checks whether a photo was taken at a given address. It uses
`GoogleMapsToolset` to validate the address, reverse geocode the photo's
coordinates, and fetch a Street View image. A sample tool,
`compare_photo_to_street_view`, then asks Gemini to compare the photo with the
Street View image and returns a verdict:

- `CONFIRMED_MATCH`
- `CONFIRMED_MISMATCH`
- `PLAUSIBLE_ALTERNATE_LOCATION`
- `INSUFFICIENT_VISUAL_CONTEXT`

The prompts treat single-family homes and multi-unit buildings differently. For
example, a parcel locker counts as a plausible location for an apartment
building but not for a house. Edit the prompts in `agent.py` for your own use
case.

## Setup

1. Enable the Street View Static, Address Validation, and Geocoding APIs on
   your Google Cloud project.
1. Set `MAPS_API_KEY`, plus the usual Gemini settings (`GOOGLE_API_KEY`, or
   `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and
   `GOOGLE_CLOUD_LOCATION`).

## Run

```shell
adk web contributing/samples/integrations
```

Select `google_maps_photo_audit`, upload a photo, and give the address and the
coordinates where the photo was taken. `SaveFilesAsArtifactsPlugin` saves the
upload as an artifact so the tool can load it.
