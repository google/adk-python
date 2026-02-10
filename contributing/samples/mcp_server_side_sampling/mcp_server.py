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

import logging
import os

from fastmcp import Context
from fastmcp import FastMCP
from fastmcp.experimental.sampling.handlers.openai import OpenAISamplingHandler
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the server's LLM handler using the OpenAI API.
# This handler will be used for all sampling requests from tools on this server.
llm_handler = OpenAISamplingHandler(
    default_model="gpt-4o",
    client=OpenAI(
        api_key=API_KEY,
    ),
)


# Create the FastMCP Server instance.
# The `sampling_handler` is configured to use the server's own LLM.
# `sampling_handler_behavior="always"` ensures the server never delegates
# sampling back to the ADK agent.
mcp = FastMCP(
    name="SentimentAnalysis",
    sampling_handler=llm_handler,
    sampling_handler_behavior="always",
)


@mcp.tool
async def analyze_sentiment(text: str, ctx: Context) -> dict:
  """Analyzes sentiment by delegating to the server's own LLM."""
  logging.info("analyze_sentiment tool called with text: %s", text)
  prompt = f"""Analyze the sentiment of the following text as positive,
                 negative, or neutral. Just output a single word.
                 Text to analyze: {text}"""

  # This delegates the LLM call to the server's own sampling handler,
  # as configured in the FastMCP instance.
  logging.info("Attempting to call ctx.sample()")
  try:
    response = await ctx.sample(prompt)
    logging.info("ctx.sample() successful. Response: %s", response)
  except Exception as e:
    logging.error("ctx.sample() failed: %s", e, exc_info=True)
    raise

  sentiment = response.text.strip().lower()

  if "positive" in sentiment:
    result = "positive"
  elif "negative" in sentiment:
    result = "negative"
  else:
    result = "neutral"

  logging.info("Sentiment analysis result: %s", result)
  return {"text": text, "sentiment": result}


if __name__ == "__main__":
  print("Starting FastMCP server with tool 'analyze_sentiment'...")
  # This runs the server process, which the ADK agent will connect to.
  mcp.run()
