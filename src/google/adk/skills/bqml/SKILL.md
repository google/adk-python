---
name: bqml
description: Machine learning in BigQuery using BQML. Create, train, evaluate, and deploy ML models directly in SQL.
license: Apache-2.0
compatibility: BigQuery, Vertex AI
metadata:
  author: Google Cloud
  version: "1.0"
  category: machine-learning
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 5
  allowed_callers:
    - bigquery_agent
    - data_science_agent
---

# BQML Skill

BigQuery ML (BQML) enables machine learning directly in BigQuery using SQL.

## When to Use

- Train ML models on BigQuery data without data movement
- Make predictions using trained models
- Evaluate model performance
- Explain predictions with feature importance

## Available Tools

| Tool | Description |
|------|-------------|
| `create_model` | Create and train a new ML model |
| `evaluate_model` | Get model evaluation metrics |
| `predict` | Make predictions with a trained model |
| `explain_predict` | Get predictions with feature attributions |
| `get_model_info` | Retrieve model metadata and training info |
| `list_models` | List models in a dataset |
| `drop_model` | Delete a model |
| `feature_importance` | Get global feature importance |
| `confusion_matrix` | Get confusion matrix for classifiers |
| `roc_curve` | Get ROC curve data for binary classifiers |

## Quick Start

1. **Create a model**: Use `create_model` with model type and training data
2. **Evaluate**: Check performance with `evaluate_model`
3. **Predict**: Use `predict` or `explain_predict` for inference

## References

- `MODEL_TYPES.md` - Supported model types and their parameters
- `BEST_PRACTICES.md` - Tips for effective BQML usage
- `SQL_EXAMPLES.md` - Common SQL patterns and examples

## Scripts

- `validate_model.py` - Validate model configuration before training
- `export_metrics.py` - Export evaluation metrics to JSON
