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

"""Setup BigQuery to Vertex AI connection for remote models."""

import argparse
import json
import sys


def generate_connection_sql(
    project_id: str,
    region: str,
    connection_name: str,
) -> str:
    """Generate SQL to create a BigQuery external connection.

    Args:
        project_id: GCP project ID
        region: Region for the connection
        connection_name: Name for the connection

    Returns:
        SQL statement to create the connection
    """
    return f"""-- Create external connection to Vertex AI
CREATE EXTERNAL CONNECTION `{project_id}.{region}.{connection_name}`
OPTIONS (
  type = 'CLOUD_RESOURCE'
);

-- Note: After creating the connection, you need to:
-- 1. Get the service account: bq show --connection {project_id}.{region}.{connection_name}
-- 2. Grant IAM role: gcloud projects add-iam-policy-binding {project_id} \\
--    --member="serviceAccount:<SERVICE_ACCOUNT>" \\
--    --role="roles/aiplatform.user"
"""


def generate_model_sql(
    project_id: str,
    dataset_id: str,
    model_name: str,
    connection_name: str,
    region: str,
    endpoint: str,
    model_type: str = "text",
) -> str:
    """Generate SQL to create a remote model.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        model_name: Name for the model
        connection_name: Connection name
        region: Region
        endpoint: Vertex AI endpoint name
        model_type: Type of model (text, embedding)

    Returns:
        SQL statement to create the model
    """
    full_connection = f"`{project_id}.{region}.{connection_name}`"
    full_model = f"`{project_id}.{dataset_id}.{model_name}`"

    if model_type == "embedding":
        return f"""-- Create remote embedding model
CREATE OR REPLACE MODEL {full_model}
REMOTE WITH CONNECTION {full_connection}
OPTIONS (
  endpoint = '{endpoint}'
);
"""
    else:
        return f"""-- Create remote text generation model
CREATE OR REPLACE MODEL {full_model}
REMOTE WITH CONNECTION {full_connection}
OPTIONS (
  endpoint = '{endpoint}',
  max_output_tokens = 1024,
  temperature = 0.2
);
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL for BigQuery-Vertex AI connection setup"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="Region for the connection",
    )
    parser.add_argument(
        "--connection-name",
        type=str,
        default="vertex_ai_connection",
        help="Name for the connection",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        help="BigQuery dataset ID (for model creation)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the remote model",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="gemini-1.5-flash",
        help="Vertex AI endpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["text", "embedding"],
        default="text",
        help="Type of model to create",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["sql", "json"],
        default="sql",
        help="Output format",
    )

    args = parser.parse_args()

    # Generate connection SQL
    connection_sql = generate_connection_sql(
        args.project_id,
        args.region,
        args.connection_name,
    )

    # Generate model SQL if dataset and model name provided
    model_sql = ""
    if args.dataset_id and args.model_name:
        model_sql = generate_model_sql(
            args.project_id,
            args.dataset_id,
            args.model_name,
            args.connection_name,
            args.region,
            args.endpoint,
            args.model_type,
        )

    if args.output == "json":
        result = {
            "connection_sql": connection_sql,
            "model_sql": model_sql if model_sql else None,
            "connection_name": f"{args.project_id}.{args.region}.{args.connection_name}",
        }
        if args.dataset_id and args.model_name:
            result["model_name"] = f"{args.project_id}.{args.dataset_id}.{args.model_name}"
        print(json.dumps(result, indent=2))
    else:
        print(connection_sql)
        if model_sql:
            print(model_sql)


if __name__ == "__main__":
    main()
