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

"""Validate BQML model configuration before training."""

import argparse
import json
import sys

VALID_MODEL_TYPES = {
    "LINEAR_REG",
    "LOGISTIC_REG",
    "KMEANS",
    "MATRIX_FACTORIZATION",
    "PCA",
    "AUTOENCODER",
    "DNN_CLASSIFIER",
    "DNN_REGRESSOR",
    "DNN_LINEAR_COMBINED_CLASSIFIER",
    "DNN_LINEAR_COMBINED_REGRESSOR",
    "BOOSTED_TREE_CLASSIFIER",
    "BOOSTED_TREE_REGRESSOR",
    "RANDOM_FOREST_CLASSIFIER",
    "RANDOM_FOREST_REGRESSOR",
    "ARIMA_PLUS",
    "TENSORFLOW",
    "ONNX",
    "XGBOOST",
}

SUPERVISED_TYPES = {
    "LINEAR_REG",
    "LOGISTIC_REG",
    "DNN_CLASSIFIER",
    "DNN_REGRESSOR",
    "DNN_LINEAR_COMBINED_CLASSIFIER",
    "DNN_LINEAR_COMBINED_REGRESSOR",
    "BOOSTED_TREE_CLASSIFIER",
    "BOOSTED_TREE_REGRESSOR",
    "RANDOM_FOREST_CLASSIFIER",
    "RANDOM_FOREST_REGRESSOR",
}


def validate_model_config(config: dict) -> dict:
    """Validate model configuration.

    Args:
        config: Model configuration dictionary with keys:
            - model_type: Type of model
            - input_label_cols: Label column(s) for supervised learning
            - features: List of feature columns
            - options: Additional model options

    Returns:
        Validation result with 'valid' boolean and 'errors'/'warnings' lists.
    """
    errors = []
    warnings = []

    # Check model type
    model_type = config.get("model_type", "").upper()
    if not model_type:
        errors.append("model_type is required")
    elif model_type not in VALID_MODEL_TYPES:
        errors.append(
            f"Invalid model_type: {model_type}. "
            f"Valid types: {', '.join(sorted(VALID_MODEL_TYPES))}"
        )

    # Check label columns for supervised learning
    if model_type in SUPERVISED_TYPES:
        label_cols = config.get("input_label_cols", [])
        if not label_cols:
            errors.append(
                f"input_label_cols required for supervised model type: {model_type}"
            )

    # Check features
    features = config.get("features", [])
    if not features:
        warnings.append("No features specified - will use all columns from query")

    # Validate options
    options = config.get("options", {})

    # Check max_iterations
    max_iter = options.get("max_iterations")
    if max_iter is not None:
        if max_iter < 1:
            errors.append("max_iterations must be >= 1")
        elif max_iter > 500:
            warnings.append(
                f"max_iterations={max_iter} is high - consider starting lower"
            )

    # Check learn_rate for boosted trees
    if "BOOSTED" in model_type:
        learn_rate = options.get("learn_rate")
        if learn_rate is not None:
            if learn_rate <= 0 or learn_rate > 1:
                errors.append("learn_rate must be in (0, 1]")
            elif learn_rate > 0.3:
                warnings.append(
                    f"learn_rate={learn_rate} is high - may cause overfitting"
                )

    # Check num_clusters for KMEANS
    if model_type == "KMEANS":
        num_clusters = options.get("num_clusters")
        if num_clusters is None:
            warnings.append("num_clusters not specified - will use default")
        elif num_clusters < 2:
            errors.append("num_clusters must be >= 2")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "model_type": model_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate BQML model configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON string or file path with model configuration",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Model type (alternative to --config)",
    )
    parser.add_argument(
        "--label-cols",
        type=str,
        nargs="+",
        help="Label columns for supervised learning",
    )

    args = parser.parse_args()

    # Build config from arguments
    if args.config:
        try:
            # Try as JSON string first
            config = json.loads(args.config)
        except json.JSONDecodeError:
            # Try as file path
            try:
                with open(args.config) as f:
                    config = json.load(f)
            except FileNotFoundError:
                print(f"Error: Config file not found: {args.config}")
                sys.exit(1)
    else:
        config = {
            "model_type": args.model_type or "",
            "input_label_cols": args.label_cols or [],
        }

    # Validate
    result = validate_model_config(config)

    # Output result
    print(json.dumps(result, indent=2))

    # Exit with error code if invalid
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
