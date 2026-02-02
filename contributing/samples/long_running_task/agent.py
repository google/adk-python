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

"""Durable session demo agent with long-running BigQuery operations.

This agent demonstrates the durable session persistence feature, which enables
checkpointing of agent state to BigQuery + GCS for recovery from failures.

To run this demo:
    1. Set up the BigQuery tables and GCS bucket (see setup.py)
    2. Set GOOGLE_CLOUD_API_KEY environment variable
    3. Run: adk web contributing/samples/long_running_task

Example prompts:
    - "Scan the bigquery-public-data.samples.shakespeare table"
    - "Get the schema of bigquery-public-data.samples.github_nested"
    - "Run a pipeline from source_table to dest_table with filter, aggregate"
"""

import os
from functools import cached_property

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.durable import BigQueryCheckpointStore
from google.adk.durable import DurableSessionConfig
from google.adk.models.google_llm import Gemini
from google.adk.tools import LongRunningFunctionTool
from google.genai import Client
from google.genai import types

from .tools import get_table_schema
from .tools import run_batch_etl_job
from .tools import run_data_pipeline
from .tools import run_demo_analysis
from .tools import run_extended_analysis
from .tools import run_ml_training_job
from .tools import simulate_long_running_scan

# Configuration
PROJECT_ID = "test-project-0728-467323"
DATASET = "adk_metadata"
GCS_BUCKET = f"{PROJECT_ID}-adk-checkpoints"

# API Key for Vertex AI (must be set via environment variable)
GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_CLOUD_API_KEY", "")


class VertexAIGemini(Gemini):
  """Custom Gemini model configured for Vertex AI with API key."""

  model: str = "gemini-3-flash-preview"

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client configured for Vertex AI."""
    return Client(
        vertexai=True,
        api_key=GOOGLE_CLOUD_API_KEY,
        http_options=types.HttpOptions(
            headers=self._tracking_headers(),
            retry_options=self.retry_options,
        ),
    )


# Create the checkpoint store
checkpoint_store = BigQueryCheckpointStore(
    project=PROJECT_ID,
    dataset=DATASET,
    gcs_bucket=GCS_BUCKET,
)

# Create the root agent with long-running tools using custom Vertex AI model
root_agent = LlmAgent(
    model=VertexAIGemini(model="gemini-3-flash-preview"),
    name="durable_bq_scanner",
    description="Long-running BigQuery scanner with durable checkpoints",
    instruction="""You are a data analyst assistant that can run various data processing jobs.

Your capabilities:
1. Get table schemas - Use get_table_schema for quick schema lookups
2. Scan tables - Use simulate_long_running_scan for table analysis (~5-10 seconds)
3. Run data pipelines - Use run_data_pipeline for multi-stage transformations
4. Demo analysis - Use run_demo_analysis for a 1-minute demo (perfect for presentations!)
5. Extended analysis - Use run_extended_analysis for jobs that run 1-60 minutes
6. ML training - Use run_ml_training_job for model training (2-30 minutes based on size)
7. Batch ETL - Use run_batch_etl_job for large ETL jobs (1-60 minutes)

For quick demos (~1 minute):
- run_demo_analysis: Specify analysis_type (e.g., "sentiment", "anomaly", "trend", "clustering")

For long-running jobs (10+ minutes):
- run_extended_analysis: Specify duration_minutes (e.g., 10, 15, 30)
- run_ml_training_job: Use dataset_size "large" (10 min), "xlarge" (15 min), or "enterprise" (30 min)
- run_batch_etl_job: Specify processing_minutes (e.g., 10, 15, 30)

The system will automatically checkpoint your progress during long-running
operations, so you can resume if interrupted.

Important: When using long-running tools, wait for them to complete before
taking further action. Do not call the same tool again if it returned a
pending status.
""",
    tools=[
        get_table_schema,
        LongRunningFunctionTool(func=simulate_long_running_scan),
        LongRunningFunctionTool(func=run_data_pipeline),
        LongRunningFunctionTool(func=run_demo_analysis),
        LongRunningFunctionTool(func=run_extended_analysis),
        LongRunningFunctionTool(func=run_ml_training_job),
        LongRunningFunctionTool(func=run_batch_etl_job),
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=1.0,  # Required for Gemini 3
    ),
)

# Create the app with durable session configuration
app = App(
    name="long_running_task",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
    durable_session_config=DurableSessionConfig(
        is_durable=True,
        checkpoint_policy="async_boundary",
        checkpoint_store=checkpoint_store,
        lease_timeout_seconds=300,
    ),
)
