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

"""BigQuery ML (BQML) Skill for ADK.

This skill provides machine learning capabilities using BigQuery ML,
allowing agents to train, evaluate, and use ML models directly in SQL.

Supported model types:
- Linear/logistic regression
- K-means clustering
- Matrix factorization
- Time series (ARIMA_PLUS)
- Deep neural networks (via Vertex AI)
- XGBoost/Random Forest
- And more

For full documentation, see:
https://cloud.google.com/bigquery/docs/bqml-introduction
"""

from __future__ import annotations

from typing import Any
from typing import List

from pydantic import Field

from ....skills.base_skill import BaseSkill
from ....skills.base_skill import SkillConfig
from ....utils.feature_decorator import experimental


@experimental
class BQMLSkill(BaseSkill):
  """Skill for BigQuery ML operations.

  This skill bundles tools for machine learning operations in BigQuery,
  including model creation, training, evaluation, and prediction.

  Supported model types:
  - LINEAR_REG: Linear regression for numeric predictions
  - LOGISTIC_REG: Logistic regression for classification
  - KMEANS: K-means clustering for segmentation
  - MATRIX_FACTORIZATION: Recommendation systems
  - ARIMA_PLUS: Time series forecasting
  - DNN_CLASSIFIER/DNN_REGRESSOR: Deep neural networks
  - BOOSTED_TREE_CLASSIFIER/BOOSTED_TREE_REGRESSOR: XGBoost models
  - RANDOM_FOREST_CLASSIFIER/RANDOM_FOREST_REGRESSOR: Random forest

  Example:
    ```python
    from google.adk.tools.bigquery.skills import BQMLSkill

    skill = BQMLSkill()
    agent = Agent(
        name="ml_analyst",
        skills=[skill],
        enable_programmatic_tool_calling=True,
    )
    ```
  """

  name: str = Field(default="bqml")
  description: str = Field(
      default=(
          "Machine learning with BigQuery ML. Train, evaluate, and deploy "
          "ML models using SQL. Supports regression, classification, "
          "clustering, time series forecasting, and deep learning models."
      )
  )
  config: SkillConfig = Field(
      default_factory=lambda: SkillConfig(
          timeout_seconds=300.0,  # ML training can take time
          max_parallel_calls=5,
      )
  )

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    """Return tool declarations for BQML operations."""
    return [
        {
            "name": "create_model",
            "description": (
                "Create and train a BigQuery ML model. "
                "Supports various model types including linear regression, "
                "logistic regression, k-means, time series, and more."
            ),
            "parameters": {
                "model_name": (
                    "Fully qualified model name (project.dataset.model)"
                ),
                "model_type": (
                    "Type of model: LINEAR_REG, LOGISTIC_REG, KMEANS, "
                    "ARIMA_PLUS, DNN_CLASSIFIER, BOOSTED_TREE_REGRESSOR, etc."
                ),
                "training_data": "SQL query or table for training data",
                "options": "Additional model options as key-value pairs",
            },
        },
        {
            "name": "evaluate_model",
            "description": (
                "Evaluate a trained model's performance. Returns metrics "
                "like accuracy, precision, recall, RMSE, MAE depending on "
                "model type."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
                "eval_data": "Optional evaluation dataset (SQL or table)",
            },
        },
        {
            "name": "predict",
            "description": (
                "Generate predictions using a trained model. "
                "Returns predicted values along with input features."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
                "input_data": "SQL query or table for prediction input",
            },
        },
        {
            "name": "explain_predict",
            "description": (
                "Generate predictions with feature attributions. "
                "Shows which features contributed most to each prediction."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
                "input_data": "SQL query or table for prediction input",
                "top_k_features": "Number of top features to show",
            },
        },
        {
            "name": "get_model_info",
            "description": (
                "Get information about a model including training stats, "
                "feature info, and hyperparameters."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
            },
        },
        {
            "name": "list_models",
            "description": "List all models in a dataset.",
            "parameters": {
                "dataset": "Fully qualified dataset (project.dataset)",
            },
        },
        {
            "name": "drop_model",
            "description": "Delete a model from the dataset.",
            "parameters": {
                "model_name": "Fully qualified model name",
            },
        },
        {
            "name": "feature_importance",
            "description": (
                "Get feature importance scores for tree-based models "
                "(XGBoost, Random Forest)."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
            },
        },
        {
            "name": "confusion_matrix",
            "description": (
                "Generate confusion matrix for classification models. "
                "Shows true/false positives and negatives."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
                "eval_data": "Optional evaluation dataset",
            },
        },
        {
            "name": "roc_curve",
            "description": (
                "Generate ROC curve data for binary classification models. "
                "Returns threshold, TPR, FPR values."
            ),
            "parameters": {
                "model_name": "Fully qualified model name",
                "eval_data": "Optional evaluation dataset",
            },
        },
    ]

  def get_orchestration_template(self) -> str:
    """Return example orchestration code for BQML operations."""
    return '''
async def train_and_evaluate(tools):
    """Train a model and evaluate its performance."""
    import asyncio

    # Create a linear regression model
    create_result = await tools.create_model(
        model_name="my_project.my_dataset.sales_predictor",
        model_type="LINEAR_REG",
        training_data="""
            SELECT
                store_id,
                product_category,
                day_of_week,
                is_holiday,
                sales_amount
            FROM `my_project.my_dataset.historical_sales`
            WHERE sales_date >= '2024-01-01'
        """,
        options={
            "input_label_cols": ["sales_amount"],
            "enable_global_explain": True
        }
    )

    if create_result.get("status") != "SUCCESS":
        return {"error": create_result.get("error_details")}

    # Evaluate and predict in parallel
    eval_result, predictions = await asyncio.gather(
        tools.evaluate_model(
            model_name="my_project.my_dataset.sales_predictor"
        ),
        tools.predict(
            model_name="my_project.my_dataset.sales_predictor",
            input_data="""
                SELECT store_id, product_category, day_of_week, is_holiday
                FROM `my_project.my_dataset.upcoming_dates`
            """
        )
    )

    return {
        "model_created": True,
        "evaluation_metrics": eval_result.get("rows", []),
        "sample_predictions": predictions.get("rows", [])[:10],
        "total_predictions": len(predictions.get("rows", []))
    }


async def cluster_customers(tools):
    """Segment customers using k-means clustering."""
    # Create k-means clustering model
    await tools.create_model(
        model_name="my_project.my_dataset.customer_segments",
        model_type="KMEANS",
        training_data="""
            SELECT
                customer_id,
                total_purchases,
                avg_order_value,
                purchase_frequency,
                days_since_last_purchase
            FROM `my_project.my_dataset.customer_metrics`
        """,
        options={
            "num_clusters": 5,
            "standardize_features": True
        }
    )

    # Get cluster assignments
    clusters = await tools.predict(
        model_name="my_project.my_dataset.customer_segments",
        input_data="SELECT * FROM `my_project.my_dataset.customer_metrics`"
    )

    # Summarize by cluster
    cluster_counts = {}
    for row in clusters.get("rows", []):
        centroid_id = row.get("CENTROID_ID")
        cluster_counts[centroid_id] = cluster_counts.get(centroid_id, 0) + 1

    return {
        "num_clusters": len(cluster_counts),
        "cluster_sizes": cluster_counts,
        "sample_assignments": clusters.get("rows", [])[:20]
    }
'''

  def filter_result(self, result: Any) -> Any:
    """Filter BQML results to reduce context size.

    - Truncates large result sets
    - Rounds numeric metrics for readability
    - Removes verbose internal fields
    """
    if not isinstance(result, dict):
      return result

    filtered = result.copy()

    # Truncate large row results
    if "rows" in filtered and len(filtered["rows"]) > 100:
      filtered["rows"] = filtered["rows"][:100]
      filtered["result_truncated"] = True
      filtered["total_rows"] = len(result["rows"])

    # Round numeric values in metrics for readability
    if "rows" in filtered:
      for row in filtered["rows"]:
        if isinstance(row, dict):
          for key, value in row.items():
            if isinstance(value, float):
              row[key] = round(value, 6)

    return filtered
