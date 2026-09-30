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

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk import Event
from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool


async def monitor_stock_price(stock_symbol: str) -> AsyncGenerator[Any, None]:
  """Starts a background monitor for the price of the given stock_symbol.

  Call this function ONLY ONCE to initiate monitoring. Once started, it runs
  continuously in the background and automatically streams price alerts.

  CRITICAL: Do NOT call this function again to "check" or "poll" for updates.
  Simply wait for the background task to yield new values and report them.
  Calling this again while running will launch a duplicate background task.
  """
  print(f"Start monitor stock price for {stock_symbol}!")

  # An Event goes straight to the user, without going over the live connection
  # to the model, so status chatter costs no model turn and cannot derail the
  # model's reasoning mid-task.
  yield Event(message=f"Connected to the {stock_symbol} price feed.")

  # A plain value goes back to the model as a FunctionResponse, and the model
  # decides how to report it. This is what a streaming tool has always done.
  await asyncio.sleep(4)
  yield f"the price for {stock_symbol} is 300"

  yield Event(message="Trading is getting busy, prices are moving fast.")

  await asyncio.sleep(4)
  yield f"the price for {stock_symbol} is 400"

  # Each yield is addressed to exactly one audience, so narrating and
  # reporting at the same moment is simply two yields.
  await asyncio.sleep(10)
  yield f"the price for {stock_symbol} is 900"
  yield Event(message=f"That is my last {stock_symbol} update for now.")


# Use this exact function to help ADK stop your streaming tools when requested.
# For example, to stop `monitor_stock_price` the model calls
# stop_streaming(function_name="monitor_stock_price").
def stop_streaming(function_name: str):
  """Stop the streaming.

  The body is intentionally empty: ADK intercepts this call and cancels the
  named tool's background task itself. Copy it as is.

  Args:
    function_name: The name of the streaming function to stop.
  """


root_agent = Agent(
    # Find supported models in Vertex here: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
    model="gemini-live-2.5-flash-native-audio",  # Vertex
    # Find supported models in Gemini API here: https://ai.google.dev/gemini-api/docs/models
    # model='gemini-2.5-flash-native-audio-preview-12-2025',  # Gemini API
    name="streaming_tool_events_agent",
    instruction="""
      You are a monitoring agent. You can monitor a stock price using
      monitor_stock_price.
      CRITICAL: Only call monitor_stock_price at most once per request. Once
      called, it runs continuously in the background. Do NOT call it again to
      "poll" or "check" for updates. Simply wait for it to stream a new price
      alert to you, and then report that alert to the user.
      If you need to stop the monitor, call stop_streaming.
      Don't ask too many questions. Don't be too talkative.
    """,
    tools=[
        monitor_stock_price,
        FunctionTool(stop_streaming),
    ],
)
