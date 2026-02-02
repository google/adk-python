#!/usr/bin/env python
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

"""Setup script for the durable session demo.

This script creates the required BigQuery dataset, tables, and GCS bucket
for the durable session persistence demo.

Usage:
    python setup.py

Prerequisites:
    - Google Cloud SDK installed and configured
    - BigQuery API enabled
    - Cloud Storage API enabled
    - Appropriate IAM permissions:
        - roles/bigquery.dataEditor
        - roles/storage.objectAdmin
"""

import argparse
import subprocess
import sys

# Configuration
PROJECT_ID = "test-project-0728-467323"
DATASET = "adk_metadata"
GCS_BUCKET = f"{PROJECT_ID}-adk-checkpoints"
LOCATION = "US"


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess:
  """Run a shell command and return the result."""
  print(f"Running: {' '.join(cmd)}")
  result = subprocess.run(cmd, capture_output=True, text=True)
  if check and result.returncode != 0:
    print(f"Error: {result.stderr}")
    if not result.stderr.strip().endswith("already exists"):
      sys.exit(1)
  return result


def create_gcs_bucket():
  """Create the GCS bucket for checkpoint blobs."""
  print("\n=== Creating GCS Bucket ===")
  run_command(
      ["gsutil", "mb", "-l", LOCATION, f"gs://{GCS_BUCKET}"], check=False
  )

  # Set lifecycle policy to delete old checkpoints after 30 days
  lifecycle_config = """
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
"""
  with open("/tmp/lifecycle.json", "w") as f:
    f.write(lifecycle_config)

  run_command(
      [
          "gsutil",
          "lifecycle",
          "set",
          "/tmp/lifecycle.json",
          f"gs://{GCS_BUCKET}",
      ],
      check=False,
  )

  print(f"GCS bucket created: gs://{GCS_BUCKET}")


def create_bigquery_dataset():
  """Create the BigQuery dataset."""
  print("\n=== Creating BigQuery Dataset ===")
  run_command(
      [
          "bq",
          "mk",
          "--dataset",
          "--location",
          LOCATION,
          f"{PROJECT_ID}:{DATASET}",
      ],
      check=False,
  )
  print(f"BigQuery dataset created: {PROJECT_ID}.{DATASET}")


def create_sessions_table():
  """Create the sessions metadata table."""
  print("\n=== Creating Sessions Table ===")

  schema = """
session_id:STRING,
status:STRING,
agent_name:STRING,
created_at:TIMESTAMP,
updated_at:TIMESTAMP,
current_checkpoint_seq:INT64,
active_lease_id:STRING,
lease_expiry:TIMESTAMP,
ttl_expiry:TIMESTAMP,
metadata:JSON
"""

  run_command(
      [
          "bq",
          "mk",
          "--table",
          f"{PROJECT_ID}:{DATASET}.sessions",
          schema.replace("\n", "").strip(),
      ],
      check=False,
  )

  print(f"Sessions table created: {PROJECT_ID}.{DATASET}.sessions")


def create_checkpoints_table():
  """Create the checkpoints table."""
  print("\n=== Creating Checkpoints Table ===")

  schema = """
session_id:STRING,
checkpoint_seq:INT64,
created_at:TIMESTAMP,
gcs_state_uri:STRING,
sha256:STRING,
size_bytes:INT64,
agent_state_json:JSON,
trigger:STRING
"""

  run_command(
      [
          "bq",
          "mk",
          "--table",
          f"{PROJECT_ID}:{DATASET}.checkpoints",
          schema.replace("\n", "").strip(),
      ],
      check=False,
  )

  print(f"Checkpoints table created: {PROJECT_ID}.{DATASET}.checkpoints")


def verify_setup():
  """Verify that all resources were created successfully."""
  print("\n=== Verifying Setup ===")

  # Check GCS bucket
  result = run_command(["gsutil", "ls", f"gs://{GCS_BUCKET}"], check=False)
  if result.returncode == 0:
    print(f"[OK] GCS bucket exists: gs://{GCS_BUCKET}")
  else:
    print(f"[FAIL] GCS bucket not found: gs://{GCS_BUCKET}")

  # Check BigQuery tables
  for table in ["sessions", "checkpoints"]:
    result = run_command(
        ["bq", "show", f"{PROJECT_ID}:{DATASET}.{table}"], check=False
    )
    if result.returncode == 0:
      print(f"[OK] BigQuery table exists: {PROJECT_ID}.{DATASET}.{table}")
    else:
      print(f"[FAIL] BigQuery table not found: {PROJECT_ID}.{DATASET}.{table}")


def cleanup():
  """Delete all resources created by this script."""
  print("\n=== Cleaning Up Resources ===")

  # Delete BigQuery tables
  for table in ["sessions", "checkpoints"]:
    run_command(
        ["bq", "rm", "-f", f"{PROJECT_ID}:{DATASET}.{table}"], check=False
    )

  # Delete BigQuery dataset
  run_command(["bq", "rm", "-f", "-d", f"{PROJECT_ID}:{DATASET}"], check=False)

  # Delete GCS bucket
  run_command(["gsutil", "rm", "-r", f"gs://{GCS_BUCKET}"], check=False)

  print("Cleanup complete.")


def main():
  parser = argparse.ArgumentParser(
      description="Setup resources for the durable session demo"
  )
  parser.add_argument(
      "--cleanup",
      action="store_true",
      help="Delete all resources instead of creating them",
  )
  parser.add_argument(
      "--verify", action="store_true", help="Only verify that resources exist"
  )
  args = parser.parse_args()

  print(f"Project: {PROJECT_ID}")
  print(f"Dataset: {DATASET}")
  print(f"GCS Bucket: {GCS_BUCKET}")
  print(f"Location: {LOCATION}")

  if args.cleanup:
    cleanup()
  elif args.verify:
    verify_setup()
  else:
    create_gcs_bucket()
    create_bigquery_dataset()
    create_sessions_table()
    create_checkpoints_table()
    verify_setup()

  print("\nDone!")


if __name__ == "__main__":
  main()
