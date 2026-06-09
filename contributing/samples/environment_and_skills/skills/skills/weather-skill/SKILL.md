---
name: weather-skill
description: A skill that provides weather information based on reference data and scripts.
metadata:
  adk_additional_tools:
    - get_wind_speed
---

Step 1: Check 'references/weather_info.md' for the current weather.
Step 2: If humidity is requested, run 'scripts/get_humidity.py' with the `location` argument.
Step 3: If wind speed is requested, use the `get_wind_speed` tool.
Step 4: Provide the complete weather update to the user.
