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

"""Run the demo agent with BigQuery Agent Analytics Plugin.

Usage:
  # Set required environment variables
  export PROJECT_ID="your-gcp-project"
  export BQ_DATASET_ID="your-bq-dataset"
  export BQ_TABLE_ID="agent_events_v2"  # optional, defaults to agent_events_v2
  export GCS_BUCKET="your-gcs-bucket"   # optional, for multimodal offloading

  # Run the demo
  python run_with_bq_analytics.py

Prerequisites:
  - BigQuery API enabled in your GCP project
  - BigQuery dataset created (table auto-created if needed)
  - IAM roles: bigquery.jobUser, bigquery.dataEditor
  - (Optional) GCS bucket for multimodal content offloading
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent import create_app_with_plugins, root_agent


def get_config_from_env() -> dict[str, str]:
  """Get configuration from environment variables."""
  project_id = os.getenv("PROJECT_ID")
  dataset_id = os.getenv("BQ_DATASET_ID")
  table_id = os.getenv("BQ_TABLE_ID", "agent_events_v2")
  gcs_bucket = os.getenv("GCS_BUCKET")

  if not project_id:
    print("ERROR: PROJECT_ID environment variable is required")
    sys.exit(1)
  if not dataset_id:
    print("ERROR: BQ_DATASET_ID environment variable is required")
    sys.exit(1)

  return {
      "project_id": project_id,
      "dataset_id": dataset_id,
      "table_id": table_id,
      "gcs_bucket": gcs_bucket,
  }


def create_bq_plugin(config: dict[str, str]) -> BigQueryAgentAnalyticsPlugin:
  """Create BigQuery Analytics Plugin with recommended configuration."""
  bq_config = BigQueryLoggerConfig(
      enabled=True,
      # Log all event types for comprehensive analytics
      event_allowlist=[
          "USER_MESSAGE_RECEIVED",
          "INVOCATION_STARTING",
          "INVOCATION_COMPLETED",
          "AGENT_STARTING",
          "AGENT_COMPLETED",
          "LLM_REQUEST",
          "LLM_RESPONSE",
          "LLM_ERROR",
          "TOOL_STARTING",
          "TOOL_COMPLETED",
          "TOOL_ERROR",
      ],
      # Content settings
      max_content_length=500 * 1024,  # 500KB per content field
      log_multi_modal_content=True,
      # Batching for efficiency
      batch_size=10,
      batch_flush_interval=2.0,
      # GCS offloading for large content (optional)
      gcs_bucket_name=config.get("gcs_bucket"),
  )

  return BigQueryAgentAnalyticsPlugin(
      project_id=config["project_id"],
      dataset_id=config["dataset_id"],
      table_id=config["table_id"],
      config=bq_config,
  )


async def run_demo():
  """Run interactive demo with BigQuery Analytics logging."""
  config = get_config_from_env()
  bq_plugin = create_bq_plugin(config)

  print("=" * 60)
  print("BigQuery Agent Analytics Demo")
  print("=" * 60)
  print(f"Project: {config['project_id']}")
  print(f"Dataset: {config['dataset_id']}")
  print(f"Table:   {config['table_id']}")
  if config.get("gcs_bucket"):
    print(f"GCS:     {config['gcs_bucket']}")
  print("=" * 60)

  # Create app with BQ plugin
  app = create_app_with_plugins(plugins=[bq_plugin])

  # Create runner and session service
  session_service = InMemorySessionService()
  runner = Runner(
      app=app,
      session_service=session_service,
  )

  # Create session
  session_id = f"bq-demo-{uuid.uuid4().hex[:8]}"
  user_id = "demo-user"
  session = await session_service.create_session(
      app_name=app.name,
      user_id=user_id,
      session_id=session_id,
  )
  print(f"Session: {session_id}")
  print("-" * 60)

  # Sample prompts to generate diverse telemetry
  demo_prompts = [
      "What's the weather like in Tokyo and Paris?",
      "Calculate a 5-day trip cost to Tokyo with mid-range hotels",
      "Make a flaky API call for 'weather-update'",
      "Analyze the sentiment of: This trip was amazing and wonderful!",
  ]

  print("\nRunning demo prompts to generate telemetry data...\n")

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

    print(f"Agent: {response_text[:200]}..." if len(response_text) > 200 else f"Agent: {response_text}")
    print("-" * 40)

  # Ensure all events are flushed
  await bq_plugin.shutdown()

  print("\n" + "=" * 60)
  print("Demo complete! Query your data with:")
  print("=" * 60)
  print(f"""
-- Token usage analysis
SELECT
  JSON_EXTRACT_SCALAR(content, '$.usage.total_token_count') as tokens,
  JSON_EXTRACT_SCALAR(content, '$.usage.prompt_token_count') as prompt_tokens,
  JSON_EXTRACT_SCALAR(content, '$.usage.candidates_token_count') as response_tokens
FROM `{config['project_id']}.{config['dataset_id']}.{config['table_id']}`
WHERE event_type = 'LLM_RESPONSE'
  AND session_id = '{session_id}';

-- Tool failure rate
SELECT
  JSON_EXTRACT_SCALAR(content, '$.tool') as tool_name,
  COUNTIF(event_type = 'TOOL_COMPLETED') as successes,
  COUNTIF(event_type = 'TOOL_ERROR') as failures,
  ROUND(100.0 * COUNTIF(event_type = 'TOOL_ERROR') /
    NULLIF(COUNT(*), 0), 2) as failure_rate_pct
FROM `{config['project_id']}.{config['dataset_id']}.{config['table_id']}`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
  AND session_id = '{session_id}'
GROUP BY tool_name;
""")


if __name__ == "__main__":
  asyncio.run(run_demo())
