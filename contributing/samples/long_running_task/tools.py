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

"""Long-running tools for the durable session demo."""

import asyncio
import random
from datetime import datetime
from typing import Any

from google.adk.tools.tool_context import ToolContext


async def simulate_long_running_scan(
    table_name: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Simulate a long-running BigQuery table scan.

  This tool demonstrates durable checkpointing by simulating a scan that
  takes several seconds. In a real scenario, this would be a BigQuery job
  that processes large amounts of data.

  Args:
    table_name: The fully-qualified BigQuery table name to scan.
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    A dictionary with scan results including status, row count, and findings.
  """
  # Simulate processing time (5-10 seconds)
  processing_time = random.uniform(5.0, 10.0)
  await asyncio.sleep(processing_time)

  # Simulate scan results
  rows_scanned = random.randint(100000, 10000000)
  findings = []

  # Generate some sample findings based on table name
  if "shakespeare" in table_name.lower():
    findings = [
        "Found 5 instances of 'to be or not to be'",
        "Most common word: 'the' (27,801 occurrences)",
        "Unique words: 29,066",
    ]
  elif "github" in table_name.lower():
    findings = [
        "Most active repository: kubernetes/kubernetes",
        "Peak commit hour: 14:00 UTC",
        "Average commits per day: 45,000",
    ]
  else:
    findings = [
        f"Scanned {rows_scanned:,} rows",
        "No anomalies detected",
        "Data quality: 99.8%",
    ]

  return {
      "status": "complete",
      "table": table_name,
      "rows_scanned": rows_scanned,
      "processing_time_seconds": round(processing_time, 2),
      "findings": findings,
  }


async def run_data_pipeline(
    source_table: str,
    destination_table: str,
    transformations: list[str],
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Run a data transformation pipeline.

  This simulates a multi-stage data pipeline that would typically be
  checkpointed at each stage for durability.

  Args:
    source_table: The source BigQuery table.
    destination_table: The destination BigQuery table.
    transformations: List of transformation operations to apply.
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    Pipeline execution results.
  """
  stages_completed = []
  total_rows_processed = 0

  # Simulate each transformation stage
  for i, transformation in enumerate(transformations):
    # Simulate stage processing time
    stage_time = random.uniform(2.0, 5.0)
    await asyncio.sleep(stage_time)

    rows_processed = random.randint(10000, 100000)
    total_rows_processed += rows_processed

    stages_completed.append({
        "stage": i + 1,
        "transformation": transformation,
        "rows_processed": rows_processed,
        "duration_seconds": round(stage_time, 2),
    })

  return {
      "status": "complete",
      "source_table": source_table,
      "destination_table": destination_table,
      "stages_completed": stages_completed,
      "total_rows_processed": total_rows_processed,
      "total_stages": len(transformations),
  }


async def run_extended_analysis(
    job_name: str,
    duration_minutes: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Run an extended analysis job for a specified duration.

  This tool simulates a long-running analysis job that can run for 10+ minutes.
  Use this to test durable checkpointing with extended job durations.

  Args:
    job_name: A descriptive name for the analysis job.
    duration_minutes: How many minutes the job should run (1-60 minutes).
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    Analysis job results with timing and metrics.
  """
  start_time = datetime.now()
  duration_seconds = min(max(duration_minutes, 1), 60) * 60

  # Process in chunks, reporting progress
  chunk_size = 30  # Report every 30 seconds
  chunks_completed = 0
  total_chunks = duration_seconds // chunk_size

  metrics = {
      "records_processed": 0,
      "anomalies_detected": 0,
      "patterns_found": 0,
  }

  for i in range(0, duration_seconds, chunk_size):
    remaining = min(chunk_size, duration_seconds - i)
    await asyncio.sleep(remaining)

    chunks_completed += 1
    metrics["records_processed"] += random.randint(100000, 500000)
    metrics["anomalies_detected"] += random.randint(0, 10)
    metrics["patterns_found"] += random.randint(1, 5)

  end_time = datetime.now()
  actual_duration = (end_time - start_time).total_seconds()

  return {
      "status": "complete",
      "job_name": job_name,
      "requested_duration_minutes": duration_minutes,
      "actual_duration_seconds": round(actual_duration, 2),
      "actual_duration_minutes": round(actual_duration / 60, 2),
      "start_time": start_time.isoformat(),
      "end_time": end_time.isoformat(),
      "metrics": metrics,
      "summary": (
          f"Processed {metrics['records_processed']:,} records, "
          f"found {metrics['anomalies_detected']} anomalies and "
          f"{metrics['patterns_found']} patterns"
      ),
  }


async def run_ml_training_job(
    model_name: str,
    dataset_size: str,
    epochs: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Run a simulated ML model training job.

  This tool simulates training a machine learning model, which can take
  10+ minutes depending on the dataset size and epochs.

  Dataset sizes and approximate training times:
  - "small": ~2 minutes
  - "medium": ~5 minutes
  - "large": ~10 minutes
  - "xlarge": ~15 minutes
  - "enterprise": ~30 minutes

  Args:
    model_name: Name for the model being trained.
    dataset_size: Size of dataset - "small", "medium", "large", "xlarge", or "enterprise".
    epochs: Number of training epochs (1-100).
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    Training results with metrics and model performance.
  """
  start_time = datetime.now()

  # Map dataset size to base training time (in seconds)
  size_to_time = {
      "small": 120,      # 2 minutes
      "medium": 300,     # 5 minutes
      "large": 600,      # 10 minutes
      "xlarge": 900,     # 15 minutes
      "enterprise": 1800,  # 30 minutes
  }

  base_time = size_to_time.get(dataset_size.lower(), 300)
  epochs = min(max(epochs, 1), 100)

  # Total time scales with epochs (but not linearly)
  total_time = base_time * (1 + (epochs - 1) * 0.1)
  total_time = min(total_time, 3600)  # Cap at 1 hour

  # Simulate training epochs
  epoch_results = []
  time_per_epoch = total_time / epochs

  for epoch in range(1, epochs + 1):
    await asyncio.sleep(time_per_epoch)

    # Simulate improving metrics over epochs
    base_loss = 2.5 - (epoch / epochs) * 2.0
    loss = base_loss + random.uniform(-0.1, 0.1)
    accuracy = min(0.5 + (epoch / epochs) * 0.45 + random.uniform(-0.02, 0.02), 0.99)

    epoch_results.append({
        "epoch": epoch,
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "learning_rate": round(0.001 * (0.95 ** (epoch - 1)), 6),
    })

  end_time = datetime.now()
  actual_duration = (end_time - start_time).total_seconds()

  final_metrics = epoch_results[-1] if epoch_results else {}

  return {
      "status": "complete",
      "model_name": model_name,
      "dataset_size": dataset_size,
      "epochs_completed": epochs,
      "start_time": start_time.isoformat(),
      "end_time": end_time.isoformat(),
      "actual_duration_seconds": round(actual_duration, 2),
      "actual_duration_minutes": round(actual_duration / 60, 2),
      "final_loss": final_metrics.get("loss"),
      "final_accuracy": final_metrics.get("accuracy"),
      "training_history": epoch_results[-5:],  # Last 5 epochs
      "model_artifact": f"gs://models/{model_name}/v1/model.pkl",
  }


async def run_batch_etl_job(
    job_id: str,
    source_tables: list[str],
    target_table: str,
    processing_minutes: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Run a batch ETL (Extract, Transform, Load) job.

  This tool simulates a large-scale ETL job that processes multiple source
  tables and loads data into a target table. Can run for 10+ minutes.

  Args:
    job_id: Unique identifier for this ETL job.
    source_tables: List of source table names to process.
    target_table: Destination table for processed data.
    processing_minutes: Estimated processing time in minutes (1-60).
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    ETL job results with detailed metrics.
  """
  start_time = datetime.now()
  duration_seconds = min(max(processing_minutes, 1), 60) * 60

  # Process each source table
  table_results = []
  time_per_table = duration_seconds / max(len(source_tables), 1)

  total_rows_extracted = 0
  total_rows_transformed = 0
  total_rows_loaded = 0

  for table in source_tables:
    await asyncio.sleep(time_per_table)

    rows_extracted = random.randint(1000000, 10000000)
    rows_transformed = int(rows_extracted * random.uniform(0.85, 0.99))
    rows_loaded = int(rows_transformed * random.uniform(0.98, 1.0))

    total_rows_extracted += rows_extracted
    total_rows_transformed += rows_transformed
    total_rows_loaded += rows_loaded

    table_results.append({
        "source_table": table,
        "rows_extracted": rows_extracted,
        "rows_transformed": rows_transformed,
        "rows_loaded": rows_loaded,
        "transform_ratio": round(rows_transformed / rows_extracted, 4),
    })

  end_time = datetime.now()
  actual_duration = (end_time - start_time).total_seconds()

  return {
      "status": "complete",
      "job_id": job_id,
      "source_tables_processed": len(source_tables),
      "target_table": target_table,
      "start_time": start_time.isoformat(),
      "end_time": end_time.isoformat(),
      "actual_duration_seconds": round(actual_duration, 2),
      "actual_duration_minutes": round(actual_duration / 60, 2),
      "total_rows_extracted": total_rows_extracted,
      "total_rows_transformed": total_rows_transformed,
      "total_rows_loaded": total_rows_loaded,
      "overall_success_rate": round(total_rows_loaded / total_rows_extracted, 4),
      "table_details": table_results,
  }


async def run_demo_analysis(
    analysis_type: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Run a 1-minute demo analysis job to showcase durable checkpointing.

  This tool is perfect for demos - it runs for exactly 1 minute with
  progress updates every 10 seconds, showing how the system handles
  long-running operations with checkpointing.

  Args:
    analysis_type: Type of analysis to run (e.g., "sentiment", "anomaly",
      "trend", "clustering").
    tool_context: The tool context for accessing state and artifacts.

  Returns:
    Analysis results with timing and metrics.
  """
  start_time = datetime.now()
  total_duration = 60  # 1 minute
  update_interval = 10  # Progress every 10 seconds

  progress_updates = []
  metrics = {
      "records_analyzed": 0,
      "insights_found": 0,
      "confidence_score": 0.0,
  }

  for i in range(0, total_duration, update_interval):
    await asyncio.sleep(update_interval)

    progress_pct = ((i + update_interval) / total_duration) * 100
    records_batch = random.randint(50000, 150000)
    metrics["records_analyzed"] += records_batch
    metrics["insights_found"] += random.randint(1, 5)
    metrics["confidence_score"] = min(
        0.6 + (progress_pct / 100) * 0.35 + random.uniform(-0.02, 0.02),
        0.99
    )

    progress_updates.append({
        "timestamp": datetime.now().isoformat(),
        "progress_percent": round(progress_pct, 1),
        "records_batch": records_batch,
        "cumulative_records": metrics["records_analyzed"],
    })

  end_time = datetime.now()
  actual_duration = (end_time - start_time).total_seconds()

  # Generate analysis-specific insights
  insights = {
      "sentiment": [
          "Overall sentiment: 72% positive",
          "Key themes: innovation, growth, sustainability",
          "Sentiment trend: improving over time",
      ],
      "anomaly": [
          "Detected 3 significant anomalies",
          "Anomaly cluster in Q3 data",
          "Root cause: seasonal variation",
      ],
      "trend": [
          "Strong upward trend detected",
          "Growth rate: 15% month-over-month",
          "Forecast: continued growth expected",
      ],
      "clustering": [
          "Identified 5 distinct clusters",
          "Largest cluster: 45% of data",
          "Cluster separation: excellent",
      ],
  }.get(analysis_type.lower(), [
      f"Completed {analysis_type} analysis",
      "Results within expected parameters",
      "No critical issues detected",
  ])

  return {
      "status": "complete",
      "analysis_type": analysis_type,
      "start_time": start_time.isoformat(),
      "end_time": end_time.isoformat(),
      "duration_seconds": round(actual_duration, 2),
      "metrics": metrics,
      "insights": insights,
      "progress_history": progress_updates,
      "summary": (
          f"Completed {analysis_type} analysis on "
          f"{metrics['records_analyzed']:,} records. "
          f"Found {metrics['insights_found']} insights with "
          f"{metrics['confidence_score']:.1%} confidence."
      ),
  }


def get_table_schema(table_name: str) -> dict[str, Any]:
  """Get the schema of a BigQuery table.

  This is a quick synchronous operation that doesn't require checkpointing.

  Args:
    table_name: The fully-qualified BigQuery table name.

  Returns:
    The table schema information.
  """
  # Simulate some common schemas
  if "shakespeare" in table_name.lower():
    return {
        "table": table_name,
        "fields": [
            {"name": "word", "type": "STRING"},
            {"name": "word_count", "type": "INTEGER"},
            {"name": "corpus", "type": "STRING"},
            {"name": "corpus_date", "type": "INTEGER"},
        ],
        "num_rows": 164656,
        "size_bytes": 6432064,
    }
  elif "github" in table_name.lower():
    return {
        "table": table_name,
        "fields": [
            {"name": "repo_name", "type": "STRING"},
            {"name": "path", "type": "STRING"},
            {"name": "content", "type": "STRING"},
            {"name": "size", "type": "INTEGER"},
        ],
        "num_rows": 2800000000,
        "size_bytes": 2500000000000,
    }
  else:
    return {
        "table": table_name,
        "fields": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "STRING"},
            {"name": "created_at", "type": "TIMESTAMP"},
        ],
        "num_rows": 1000000,
        "size_bytes": 100000000,
    }
