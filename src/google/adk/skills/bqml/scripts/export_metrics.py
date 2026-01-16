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

"""Export BQML model evaluation metrics to JSON format."""

import argparse
from datetime import datetime
import json
import sys


def format_metrics(raw_metrics: dict, model_name: str = None) -> dict:
    """Format evaluation metrics for export.

    Args:
        raw_metrics: Raw metrics from ML.EVALUATE
        model_name: Optional model name to include

    Returns:
        Formatted metrics dictionary
    """
    formatted = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "metrics": {},
    }

    # Classification metrics
    classification_keys = [
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "log_loss",
        "roc_auc",
    ]

    # Regression metrics
    regression_keys = [
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "r2_score",
        "explained_variance",
    ]

    # Clustering metrics
    clustering_keys = [
        "davies_bouldin_index",
        "mean_squared_distance",
    ]

    all_keys = classification_keys + regression_keys + clustering_keys

    for key in all_keys:
        if key in raw_metrics:
            value = raw_metrics[key]
            # Round floats for readability
            if isinstance(value, float):
                value = round(value, 6)
            formatted["metrics"][key] = value

    # Determine model type from available metrics
    if any(k in raw_metrics for k in ["roc_auc", "precision", "recall"]):
        formatted["model_category"] = "classification"
    elif any(k in raw_metrics for k in ["mean_squared_error", "r2_score"]):
        formatted["model_category"] = "regression"
    elif any(k in raw_metrics for k in ["davies_bouldin_index"]):
        formatted["model_category"] = "clustering"
    else:
        formatted["model_category"] = "unknown"

    return formatted


def summarize_metrics(metrics: dict) -> str:
    """Create a human-readable summary of metrics.

    Args:
        metrics: Formatted metrics dictionary

    Returns:
        Summary string
    """
    lines = []
    lines.append(f"Model: {metrics.get('model_name', 'Unknown')}")
    lines.append(f"Category: {metrics.get('model_category', 'Unknown')}")
    lines.append(f"Evaluated: {metrics.get('timestamp', 'Unknown')}")
    lines.append("")

    m = metrics.get("metrics", {})

    if metrics.get("model_category") == "classification":
        lines.append("Classification Metrics:")
        if "accuracy" in m:
            lines.append(f"  Accuracy:  {m['accuracy']:.4f}")
        if "precision" in m:
            lines.append(f"  Precision: {m['precision']:.4f}")
        if "recall" in m:
            lines.append(f"  Recall:    {m['recall']:.4f}")
        if "f1_score" in m:
            lines.append(f"  F1 Score:  {m['f1_score']:.4f}")
        if "roc_auc" in m:
            lines.append(f"  ROC AUC:   {m['roc_auc']:.4f}")

    elif metrics.get("model_category") == "regression":
        lines.append("Regression Metrics:")
        if "r2_score" in m:
            lines.append(f"  R² Score: {m['r2_score']:.4f}")
        if "mean_absolute_error" in m:
            lines.append(f"  MAE:      {m['mean_absolute_error']:.4f}")
        if "mean_squared_error" in m:
            lines.append(f"  MSE:      {m['mean_squared_error']:.4f}")

    elif metrics.get("model_category") == "clustering":
        lines.append("Clustering Metrics:")
        if "davies_bouldin_index" in m:
            lines.append(f"  Davies-Bouldin Index: {m['davies_bouldin_index']:.4f}")
        if "mean_squared_distance" in m:
            lines.append(f"  Mean Squared Distance: {m['mean_squared_distance']:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Export BQML evaluation metrics to JSON"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="JSON string with raw metrics from ML.EVALUATE",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to include in output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary instead of JSON",
    )

    args = parser.parse_args()

    try:
        raw_metrics = json.loads(args.metrics)
    except json.JSONDecodeError as e:
        print(f"Error parsing metrics JSON: {e}", file=sys.stderr)
        sys.exit(1)

    formatted = format_metrics(raw_metrics, args.model_name)

    if args.summary:
        output = summarize_metrics(formatted)
    else:
        output = json.dumps(formatted, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Metrics exported to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
