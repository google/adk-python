#!/usr/bin/env python3
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

"""Run the demo agent with Cloud Logging via OpenTelemetry.

This script generates telemetry data via Cloud Logging for comparison
with the BigQuery Agent Analytics Plugin approach.
"""

from __future__ import annotations

import asyncio
import os
import uuid

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project-0728-467323")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("VERTEXAI_PROJECT", "test-project-0728-467323")
os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("OTEL_SERVICE_NAME", "bq_vs_cloud_logging_demo")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.telemetry.google_cloud import get_gcp_exporters, get_gcp_resource
from google.adk.telemetry.setup import maybe_set_otel_providers
from google.genai import types

from agent import create_app_with_plugins, root_agent

PROJECT_ID = "test-project-0728-467323"


async def run_demo():
  """Run interactive demo with Cloud Logging."""
  # Set up OTel with Cloud Trace and Cloud Logging
  print("Setting up OpenTelemetry with Cloud Trace and Cloud Logging...")
  otel_hooks = get_gcp_exporters(
      enable_cloud_tracing=True,  # Enabled - requires Telemetry API
      enable_cloud_logging=True,
  )
  # Get GCP resource with project_id to satisfy Cloud Trace requirements
  gcp_resource = get_gcp_resource(project_id=PROJECT_ID)
  maybe_set_otel_providers([otel_hooks], otel_resource=gcp_resource)

  print("=" * 60)
  print("Cloud Logging Demo (via OpenTelemetry)")
  print("=" * 60)
  print(f"Project: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
  print("=" * 60)

  # Create app without BQ plugin
  app = create_app_with_plugins()

  # Create runner and session service
  session_service = InMemorySessionService()
  runner = Runner(
      app=app,
      session_service=session_service,
  )

  # Create session
  session_id = f"cloud-logging-demo-{uuid.uuid4().hex[:8]}"
  user_id = "demo-user"
  session = await session_service.create_session(
      app_name=app.name,
      user_id=user_id,
      session_id=session_id,
  )
  print(f"Session: {session_id}")
  print("-" * 60)

  # Sample prompts
  demo_prompts = [
      "What's the weather like in Tokyo?",
      "Calculate a 3-day trip cost to Paris with luxury hotels",
      "Make a flaky API call for 'data-sync'",
      "Analyze the sentiment of: This experience was fantastic!",
  ]

  print("\nRunning demo prompts with Cloud Logging...\n")

  for prompt in demo_prompts:
    print(f"User: {prompt}")
    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
      if hasattr(event, "content") and event.content:
        for part in event.content.parts or []:
          if hasattr(part, "text") and part.text:
            response_text += part.text

    if len(response_text) > 150:
      print(f"Agent: {response_text[:150]}...")
    else:
      print(f"Agent: {response_text}")
    print("-" * 40)

  print("\n" + "=" * 60)
  print("Cloud Logging Demo Complete!")
  print("=" * 60)
  print(f"Session ID: {session_id}")
  print("\nLogs should appear in Cloud Logging within seconds.")
  print("\nView logs at:")
  print(
      f"  https://console.cloud.google.com/logs/query;query=logName%3D%22projects%2F{os.environ.get('GOOGLE_CLOUD_PROJECT')}%2Flogs%2Fadk-otel%22"
  )


if __name__ == "__main__":
  asyncio.run(run_demo())
