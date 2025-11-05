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

from datetime import datetime
import random
import time
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def get_weather(location: str, tool_context: ToolContext) -> Dict[str, Any]:
  """Get weather information for a location.
  Args:
    location: The city or location to get weather for.
  Returns:
    A dictionary containing weather information.
  """
  # Simulate weather data
  temperatures = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
  conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]

  return {
      "location": location,
      "temperature": random.choice(temperatures),
      "condition": random.choice(conditions),
      "humidity": random.randint(30, 90),
      "timestamp": datetime.now().isoformat(),
  }


async def calculate_async(operation: str, x: float, y: float) -> Dict[str, Any]:
  """Perform async mathematical calculations.
  Args:
    operation: The operation to perform (add, subtract, multiply, divide).
    x: First number.
    y: Second number.
  Returns:
    A dictionary containing the calculation result.
  """
  # Simulate some async work
  await asyncio.sleep(0.1)

  operations = {
      "add": x + y,
      "subtract": x - y,
      "multiply": x * y,
      "divide": x / y if y != 0 else float("inf"),
  }

  result = operations.get(operation.lower(), "Unknown operation")

  return {
      "operation": operation,
      "x": x,
      "y": y,
      "result": result,
      "timestamp": datetime.now().isoformat(),
  }


def log_activity(message: str, tool_context: ToolContext) -> Dict[str, str]:
  """Log an activity message with timestamp.
  Args:
    message: The message to log.
  Returns:
    A dictionary confirming the log entry.
  """
  if "activity_log" not in tool_context.state:
    tool_context.state["activity_log"] = []

  log_entry = {"timestamp": datetime.now().isoformat(), "message": message}
  tool_context.state["activity_log"].append(log_entry)

  return {
      "status": "logged",
      "entry": log_entry,
      "total_entries": len(tool_context.state["activity_log"]),
  }


# Before tool callbacks
def before_tool_audit_callback(
    tool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
  """Audit callback that logs all tool calls before execution."""
  print(f"🔍 AUDIT: About to call tool '{tool.name}' with args: {args}")

  # Add audit info to tool context state
  if "audit_log" not in tool_context.state:
    tool_context.state["audit_log"] = []

  tool_context.state["audit_log"].append({
      "type": "before_call",
      "tool_name": tool.name,
      "args": args,
      "timestamp": datetime.now().isoformat(),
  })

  # Return None to allow normal tool execution
  return None


def before_tool_security_callback(
    tool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
  """Security callback that can block certain tool calls."""
  # Example: Block weather requests for restricted locations
  if tool.name == "get_weather" and args.get("location", "").lower() in [
      "classified",
      "secret",
  ]:
    print(
        "🚫 SECURITY: Blocked weather request for restricted location:"
        f" {args.get('location')}"
    )
    return {
        "error": "Access denied",
        "reason": "Location access is restricted",
        "requested_location": args.get("location"),
    }

  # Allow other calls to proceed
  return None


async def before_tool_async_callback(
    tool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
  """Async before callback that can add preprocessing."""
  print(f"⚡ ASYNC BEFORE: Processing tool '{tool.name}' asynchronously")

  # Simulate some async preprocessing
  await asyncio.sleep(0.05)

  # For calculation tool, we could add validation
  if (
      tool.name == "calculate_async"
      and args.get("operation") == "divide"
      and args.get("y") == 0
  ):
    print("🚫 VALIDATION: Prevented division by zero")
    return {
        "error": "Division by zero",
        "operation": args.get("operation"),
        "x": args.get("x"),
        "y": args.get("y"),
    }

  return None


# After tool callbacks
def after_tool_enhancement_callback(
    tool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
  """Enhance tool responses with additional metadata."""
  print(f"✨ ENHANCE: Adding metadata to response from '{tool.name}'")

  # Add enhancement metadata
  enhanced_response = tool_response.copy()
  enhanced_response.update({
      "enhanced": True,
      "enhancement_timestamp": datetime.now().isoformat(),
      "tool_name": tool.name,
      "execution_context": "live_streaming",
  })

  return enhanced_response


async def after_tool_async_callback(
    tool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
  """Async after callback for post-processing."""
  print(
      f"🔄 ASYNC AFTER: Post-processing response from '{tool.name}'"
      " asynchronously"
  )

  # Simulate async post-processing
  await asyncio.sleep(0.05)

  # Add async processing metadata
  processed_response = tool_response.copy()
  processed_response.update({
      "async_processed": True,
      "processing_time": "0.05s",
      "processor": "async_after_callback",
  })

  return processed_response


import asyncio

# Create the agent with tool callbacks
root_agent = Agent(
    # find supported models here: https://google.github.io/adk-docs/get-started/streaming/quickstart-streaming/
    model="gemini-2.0-flash-live-preview-04-09",  # for Vertex project
    # model="gemini-live-2.5-flash-preview",  # for AI studio key
    name="tool_callbacks_agent",
    description=(
        "Live streaming agent that demonstrates tool callbacks functionality. "
        "It can get weather, perform calculations, and log activities while "
        "showing how before and after tool callbacks work in live mode."
    ),
    instruction="""
      You are a helpful assistant that can:
      1. Get weather information for any location using the get_weather tool
      2. Perform mathematical calculations using the calculate_async tool
      3. Log activities using the log_activity tool

      Important behavioral notes:
      - You have several callbacks that will be triggered before and after tool calls
      - Before callbacks can audit, validate, or even block tool calls
      - After callbacks can enhance or modify tool responses
      - Some locations like "classified" or "secret" are restricted for weather requests
      - Division by zero will be prevented by validation callbacks
      - All your tool responses will be enhanced with additional metadata

      When users ask you to test callbacks, explain what's happening with the callback system.
      Be conversational and explain the callback behavior you observe.
    """,
    tools=[
        get_weather,
        calculate_async,
        log_activity,
    ],
    # Multiple before tool callbacks (will be processed in order until one returns a response)
    before_tool_callback=[
        before_tool_audit_callback,
        before_tool_security_callback,
        before_tool_async_callback,
    ],
    # Multiple after tool callbacks (will be processed in order until one returns a response)
    after_tool_callback=[
        after_tool_enhancement_callback,
        after_tool_async_callback,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)
