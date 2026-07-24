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

"""A small home-automation agent shared by the evaluation samples.

All tools are deterministic (backed by in-memory dicts) so that eval
trajectories are reproducible. `reset_data()` is picked up by `adk eval` to
reset state between eval cases.
"""

from google.adk import Agent

_DEVICES = None
_TEMPERATURES = None


def reset_data() -> None:
  """Resets in-memory state. Called by adk eval between eval cases."""
  global _DEVICES, _TEMPERATURES
  _DEVICES = {
      "device_1": {"status": "ON", "location": "Living Room"},
      "device_2": {"status": "OFF", "location": "Bedroom"},
      "device_3": {"status": "OFF", "location": "Kitchen"},
  }
  _TEMPERATURES = {"Living Room": 22, "Bedroom": 20, "Kitchen": 24}


# Initialize module-level state at import time.
reset_data()


def get_device_info(device_id: str) -> dict:
  """Returns the status and location of a device, or an error string."""
  return _DEVICES.get(device_id, {"error": "Device not found"})


def set_device_info(device_id: str, status: str) -> str:
  """Sets a device status to 'ON' or 'OFF'."""
  if device_id not in _DEVICES:
    return "Device not found"
  _DEVICES[device_id]["status"] = status
  return f"Device {device_id} is now {status}."


def get_temperature(location: str) -> str:
  """Returns the current temperature (Celsius) of a location."""
  if location not in _TEMPERATURES:
    return "Location not found"
  return f"{_TEMPERATURES[location]}"


def set_temperature(location: str, temperature: int) -> str:
  """Sets the target temperature (Celsius) for a location.

  Acceptable range is 18-30 Celsius. Do not call this tool with a value
  outside that range.
  """
  if location not in _TEMPERATURES:
    return "Location not found"
  _TEMPERATURES[location] = temperature
  return f"Temperature in {location} set to {temperature}C."


def list_devices(status: str = "", location: str = "") -> list:
  """Lists devices, optionally filtered by status and/or location."""
  result = []
  for device_id, info in _DEVICES.items():
    if (not status or info["status"] == status) and (
        not location or info["location"] == location
    ):
      result.append({"device_id": device_id, **info})
  return result


root_agent = Agent(
    name="home_automation_agent",
    description="Controls smart-home devices and temperature.",
    instruction=(
        "You are a home-automation assistant. Use the available tools to"
        " inspect and control devices and temperatures. When the user asks to"
        " change something, call the matching tool, then confirm the result in"
        " one short sentence. If the user asks to set a temperature outside the"
        " safe range of 18-30 Celsius, refuse and do not call the tool."
    ),
    tools=[
        get_device_info,
        set_device_info,
        get_temperature,
        set_temperature,
        list_devices,
    ],
)
