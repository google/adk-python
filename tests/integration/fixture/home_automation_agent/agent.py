# Copyright 2025 Google LLC
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

import os
import sys

from google.adk import Agent

DEVICE_DB = {
    "device_1": {"status": "ON", "location": "Living Room"},
    "device_2": {"status": "OFF", "location": "Bedroom"},
    "device_3": {"status": "OFF", "location": "Kitchen"},
}

TEMPERATURE_DB = {
    "Living Room": 22,
    "Bedroom": 20,
    "Kitchen": 24,
}

SCHEDULE_DB = {
    "device_1": {"time": "18:00", "status": "ON"},
    "device_2": {"time": "22:00", "status": "OFF"},
}

USER_PREFERENCES_DB = {
    "user_x": {"preferred_temp": 21, "location": "Bedroom"},
    "user_x": {"preferred_temp": 21, "location": "Living Room"},
    "user_y": {"preferred_temp": 23, "location": "Living Room"},
}


def reset_data():
  global DEVICE_DB
  global TEMPERATURE_DB
  global SCHEDULE_DB
  global USER_PREFERENCES_DB
  DEVICE_DB = {
      "device_1": {"status": "ON", "location": "Living Room"},
      "device_2": {"status": "OFF", "location": "Bedroom"},
      "device_3": {"status": "OFF", "location": "Kitchen"},
  }

  TEMPERATURE_DB = {
      "Living Room": 22,
      "Bedroom": 20,
      "Kitchen": 24,
  }

  SCHEDULE_DB = {
      "device_1": {"time": "18:00", "status": "ON"},
      "device_2": {"time": "22:00", "status": "OFF"},
  }

  USER_PREFERENCES_DB = {
      "user_x": {"preferred_temp": 21, "location": "Bedroom"},
      "user_x": {"preferred_temp": 21, "location": "Living Room"},
      "user_y": {"preferred_temp": 23, "location": "Living Room"},
  }


def get_device_info(device_id: str) -> dict:
  """Get the current status and location of a AC device.

  Args:
      device_id (str): The unique identifier of the device.

  Returns:
      dict: A dictionary containing the following fields, or 'Device not found'
      if the device_id does not exist:
        - status: The current status of the device (e.g., 'ON', 'OFF')
        - location: The location where the device is installed (e.g., 'Living
        Room', 'Bedroom', ''Kitchen')
  """
  return DEVICE_DB.get(device_id, "Device not found")


# def set_device_info(device_id: str, updates: dict) -> str:
# """Update the information of a AC device, specifically its status and/or location.

# Args:
#     device_id (str): Required. The unique identifier of the device.
#     updates (dict): Required. A dictionary containing the fields to be
#       updated. Supported keys: - "status" (str): The new status to set for the
#       device. Accepted values: 'ON', 'OFF'. **Only these values are allowed.**
#       - "location" (str): The new location to set for the device. Accepted
#       values: 'Living Room', 'Bedroom', 'Kitchen'. **Only these values are
#         allowed.**


# Returns:
#     str: A message indicating whether the device information was successfully
#     updated.
# """
# if device_id in DEVICE_DB:
#   if "status" in updates:
#     DEVICE_DB[device_id]["status"] = updates["status"]
#   if "location" in updates:
#     DEVICE_DB[device_id]["location"] = updates["location"]
#   return f"Device {device_id} information updated: {updates}."
# return "Device not found"
def set_device_info(
    device_id: str, status: str = "", location: str = ""
) -> str:
  """Update the information of a AC device, specifically its status and/or location.

  Args:
      device_id (str): Required. The unique identifier of the device.
      status (str): The new status to set for the
        device. Accepted values: 'ON', 'OFF'. **Only these values are allowed.**
      location (str): The new location to set for the device. Accepted
        values: 'Living Room', 'Bedroom', 'Kitchen'. **Only these values are
          allowed.**

  Returns:
      str: A message indicating whether the device information was successfully
      updated.
  """
  if device_id in DEVICE_DB:
    if status:
      DEVICE_DB[device_id]["status"] = status
      return f"Device {device_id} information updated: status -> {status}."
    if location:
      DEVICE_DB[device_id]["location"] = location
      return f"Device {device_id} information updated: location -> {location}."
  return "Device not found"


def get_temperature(location: str) -> int:
  """Get the current temperature in celsius of a location (e.g., 'Living Room', 'Bedroom', 'Kitchen').

  Args:
      location (str): The location for which to retrieve the temperature (e.g.,
        'Living Room', 'Bedroom', 'Kitchen').

  Returns:
      int: The current temperature in celsius in the specified location, or
      'Location not found' if the location does not exist.
  """
  return TEMPERATURE_DB.get(location, "Location not found")


def set_temperature(location: str, temperature: int) -> str:
  """Set the desired temperature in celsius for a location.

  Acceptable range of temperature: 18-30 celsius. If it's out of the range, do
  not call this tool.

  Args:
      location (str): The location where the temperature should be set.
      temperature (int): The desired temperature as integer to set in celsius.
        Acceptable range: 18-30 celsius.

  Returns:
      str: A message indicating whether the temperature was successfully set.
  """
  if location in TEMPERATURE_DB:
    TEMPERATURE_DB[location] = temperature
    return f"Temperature in {location} set to {temperature}°C."
  return "Location not found"


def get_user_preferences(user_id: str) -> dict:
  """Get the temperature preferences and preferred location of a user_id.

  user_id must be provided.

  Args:
      user_id (str): The unique identifier of the user.

  Returns:
      dict: A dictionary containing the following fields, or 'User not found' if
      the user_id does not exist:
        - preferred_temp: The user's preferred temperature.
        - location: The location where the user prefers to be.
  """
  return USER_PREFERENCES_DB.get(user_id, "User not found")


def set_device_schedule(device_id: str, time: str, status: str) -> str:
  """Schedule a device to change its status at a specific time.

  Args:
      device_id (str): The unique identifier of the device.
      time (str): The time at which the device should change its status (format:
        'HH:MM').
      status (str): The status to set for the device at the specified time
        (e.g., 'ON', 'OFF').

  Returns:
      str: A message indicating whether the schedule was successfully set.
  """
  if device_id in DEVICE_DB:
    SCHEDULE_DB[device_id] = {"time": time, "status": status}
    return f"Device {device_id} scheduled to turn {status} at {time}."
  return "Device not found"


def get_device_schedule(device_id: str) -> dict:
  """Retrieve the schedule of a device.

  Args:
      device_id (str): The unique identifier of the device.

  Returns:
      dict: A dictionary containing the following fields, or 'Schedule not
      found' if the device_id does not exist:
        - time: The scheduled time for the device to change its status (format:
        'HH:MM').
        - status: The status that will be set at the scheduled time (e.g., 'ON',
        'OFF').
  """
  return SCHEDULE_DB.get(device_id, "Schedule not found")


def celsius_to_fahrenheit(celsius: int) -> float:
  """Convert Celsius to Fahrenheit.

  You must call this to do the conversion of temperature, so you can get the
  precise number in required format.

  Args:
      celsius (int): Temperature in Celsius.

  Returns:
      float: Temperature in Fahrenheit.
  """
  return (celsius * 9 / 5) + 32


def fahrenheit_to_celsius(fahrenheit: float) -> int:
  """Convert Fahrenheit to Celsius.

  You must call this to do the conversion of temperature, so you can get the
  precise number in required format.

  Args:
      fahrenheit (float): Temperature in Fahrenheit.

  Returns:
      int: Temperature in Celsius.
  """
  return int((fahrenheit - 32) * 5 / 9)


def list_devices(status: str = "", location: str = "") -> list:
  """Retrieve a list of AC devices, filtered by status and/or location when provided.

  For cost efficiency, always apply as many filters (status and location) as
  available in the input arguments.

  Args:
      status (str, optional): The status to filter devices by (e.g., 'ON',
        'OFF'). Defaults to None.
      location (str, optional): The location to filter devices by (e.g., 'Living
        Room', 'Bedroom', ''Kitchen'). Defaults to None.

  Returns:
      list: A list of dictionaries, each containing the device ID, status, and
      location, or an empty list if no devices match the criteria.
  """
  devices = []
  for device_id, info in DEVICE_DB.items():
    if ((not status) or info["status"] == status) and (
        (not location) or info["location"] == location
    ):
      devices.append({
          "device_id": device_id,
          "status": info["status"],
          "location": info["location"],
      })
  return devices if devices else "No devices found matching the criteria."


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="Home_automation_agent",
    instruction="""
    You are Home Automation Agent. You are responsible for controlling the devices in the home.
    """,
    tools=[
        get_device_info,
        set_device_info,
        get_temperature,
        set_temperature,
        get_user_preferences,
        set_device_schedule,
        get_device_schedule,
        celsius_to_fahrenheit,
        fahrenheit_to_celsius,
        list_devices,
    ],
)
