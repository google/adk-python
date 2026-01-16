#!/usr/bin/env python3
"""Setup remote model connections in BigQuery for AI operations.

This script creates BigQuery remote models connected to Vertex AI endpoints
for text generation, embeddings, and other AI operations.

Usage:
    python setup_remote_model.py --project-id PROJECT --model-type text-generation
    python setup_remote_model.py --project-id PROJECT --model-type embeddings
    python setup_remote_model.py --project-id PROJECT --endpoint gemini-2.0-flash

Examples:
    # Create Gemini model for text generation
    python setup_remote_model.py \\
        --project-id my-project \\
        --dataset my_dataset \\
        --model-name gemini_model \\
        --endpoint gemini-2.0-flash

    # Create embedding model
    python setup_remote_model.py \\
        --project-id my-project \\
        --dataset my_dataset \\
        --model-name embedding_model \\
        --endpoint text-embedding-005

    # Use specific connection
    python setup_remote_model.py \\
        --project-id my-project \\
        --dataset my_dataset \\
        --model-name my_model \\
        --endpoint gemini-1.5-pro \\
        --connection my-project.us.my-connection
"""

import argparse
import json
import sys
from typing import Optional

# Model presets for common use cases
MODEL_PRESETS = {
    "text-generation": {
        "endpoint": "gemini-2.0-flash",
        "description": "Fast text generation with Gemini 2.0 Flash",
    },
    "text-generation-pro": {
        "endpoint": "gemini-1.5-pro",
        "description": "High-quality text generation with Gemini 1.5 Pro",
    },
    "embeddings": {
        "endpoint": "text-embedding-005",
        "description": "Text embeddings for semantic search",
    },
    "embeddings-multilingual": {
        "endpoint": "text-multilingual-embedding-002",
        "description": "Multilingual text embeddings",
    },
    "multimodal-embeddings": {
        "endpoint": "multimodalembedding@001",
        "description": "Multimodal embeddings for text, images, and video",
    },
    "claude-sonnet": {
        "endpoint": "claude-3-5-sonnet@20241022",
        "description": "Anthropic Claude 3.5 Sonnet",
    },
    "llama": {
        "endpoint": "llama-3.1-70b-instruct-maas",
        "description": "Meta Llama 3.1 70B",
    },
}


def generate_create_model_sql(
    project_id: str,
    dataset: str,
    model_name: str,
    endpoint: str,
    connection: Optional[str] = None,
    replace: bool = True,
) -> str:
  """Generate CREATE MODEL SQL statement."""
  replace_clause = "OR REPLACE " if replace else ""
  connection_clause = f"`{connection}`" if connection else "DEFAULT"

  sql = f"""CREATE {replace_clause}MODEL `{project_id}.{dataset}.{model_name}`
  REMOTE WITH CONNECTION {connection_clause}
  OPTIONS (ENDPOINT = '{endpoint}');"""

  return sql


def list_presets() -> None:
  """Print available model presets."""
  print("\nAvailable model presets:")
  print("-" * 60)
  for name, config in MODEL_PRESETS.items():
    print(f"  {name:25} - {config['description']}")
    print(f"  {'':25}   Endpoint: {config['endpoint']}")
  print("-" * 60)


def main():
  parser = argparse.ArgumentParser(
      description="Setup BigQuery remote models for AI operations",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )

  parser.add_argument(
      "--project-id",
      required=True,
      help="Google Cloud project ID",
  )
  parser.add_argument(
      "--dataset",
      default="ai_models",
      help="BigQuery dataset name (default: ai_models)",
  )
  parser.add_argument(
      "--model-name",
      help="Name for the model (auto-generated if not specified)",
  )
  parser.add_argument(
      "--endpoint",
      help="Vertex AI endpoint (e.g., gemini-2.0-flash, text-embedding-005)",
  )
  parser.add_argument(
      "--model-type",
      choices=list(MODEL_PRESETS.keys()),
      help="Use a preset model type",
  )
  parser.add_argument(
      "--connection",
      help="BigQuery connection ID (uses DEFAULT if not specified)",
  )
  parser.add_argument(
      "--region",
      default="us",
      help="Region for connection (default: us)",
  )
  parser.add_argument(
      "--list-presets",
      action="store_true",
      help="List available model presets",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print SQL without executing",
  )
  parser.add_argument(
      "--output",
      choices=["sql", "json"],
      default="sql",
      help="Output format (default: sql)",
  )

  args = parser.parse_args()

  if args.list_presets:
    list_presets()
    return 0

  # Resolve endpoint from preset or direct specification
  if args.model_type:
    preset = MODEL_PRESETS[args.model_type]
    endpoint = preset["endpoint"]
    default_model_name = args.model_type.replace("-", "_") + "_model"
  elif args.endpoint:
    endpoint = args.endpoint
    default_model_name = (
        endpoint.replace("-", "_").replace("@", "_").replace(".", "_")
    )
  else:
    print("Error: Either --model-type or --endpoint must be specified")
    print("Use --list-presets to see available presets")
    return 1

  model_name = args.model_name or default_model_name

  # Generate SQL
  sql = generate_create_model_sql(
      project_id=args.project_id,
      dataset=args.dataset,
      model_name=model_name,
      endpoint=endpoint,
      connection=args.connection,
      replace=True,
  )

  if args.output == "json":
    result = {
        "project_id": args.project_id,
        "dataset": args.dataset,
        "model_name": model_name,
        "endpoint": endpoint,
        "connection": args.connection or "DEFAULT",
        "sql": sql,
        "dry_run": args.dry_run,
    }
    print(json.dumps(result, indent=2))
  else:
    print(f"\n-- Create remote model: {model_name}")
    print(f"-- Endpoint: {endpoint}")
    print(f"-- Connection: {args.connection or 'DEFAULT'}")
    print()
    print(sql)
    print()

  if args.dry_run:
    print("-- Dry run mode: SQL not executed")
    return 0

  # Execute the SQL
  try:
    from google.cloud import bigquery

    client = bigquery.Client(project=args.project_id)

    print(f"Creating model {args.project_id}.{args.dataset}.{model_name}...")
    query_job = client.query(sql)
    query_job.result()  # Wait for completion

    print(f"Successfully created model: {model_name}")

    # Verify the model
    model_ref = f"{args.project_id}.{args.dataset}.{model_name}"
    model = client.get_model(model_ref)
    print(f"Model type: {model.model_type}")
    print(f"Created: {model.created}")

    return 0

  except ImportError:
    print("\n-- Note: google-cloud-bigquery not installed")
    print("-- To execute, install it with: pip install google-cloud-bigquery")
    print("-- Or run the SQL manually in BigQuery Console")
    return 0

  except Exception as e:
    print(f"Error creating model: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
