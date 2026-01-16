---
name: bqml
description: Train and deploy traditional ML models in BigQuery using SQL - classification, regression, clustering, time series forecasting, and recommendations. Use when building predictive models on BigQuery data without data movement.
license: Apache-2.0
compatibility: BigQuery
metadata:
  author: Google Cloud
  version: "2.0"
  category: machine-learning
adk:
  config:
    timeout_seconds: 600
    max_parallel_calls: 3
  allowed_callers:
    - bigquery_agent
    - data_science_agent
    - ml_agent
---

# BQML Skill

BigQuery ML (BQML) enables training, evaluating, and deploying machine learning models directly in BigQuery using SQL. No data movement required.

## When to Use This Skill

Use BQML when you need to:
- Train classification models (predict categories)
- Train regression models (predict numeric values)
- Build time series forecasting models
- Create clustering/segmentation models
- Build recommendation systems
- Detect anomalies in data
- Import and deploy external models (TensorFlow, ONNX, XGBoost)

**Note**: For generative AI tasks (text generation, embeddings, semantic search), use the `bigquery-ai` skill instead.

## Supported Model Types

| Category | Model Types | Use Cases |
|----------|-------------|-----------|
| **Classification** | LOGISTIC_REG, BOOSTED_TREE_CLASSIFIER, RANDOM_FOREST_CLASSIFIER, DNN_CLASSIFIER | Churn prediction, fraud detection, sentiment |
| **Regression** | LINEAR_REG, BOOSTED_TREE_REGRESSOR, RANDOM_FOREST_REGRESSOR, DNN_REGRESSOR | Price prediction, demand forecasting |
| **Clustering** | KMEANS | Customer segmentation, anomaly detection |
| **Time Series** | ARIMA_PLUS, ARIMA_PLUS_XREG | Sales forecasting, demand planning |
| **Recommendations** | MATRIX_FACTORIZATION | Product recommendations, content suggestions |
| **Dimensionality Reduction** | PCA, AUTOENCODER | Feature engineering, anomaly detection |
| **Imported Models** | TENSORFLOW, ONNX, XGBOOST | Deploy pre-trained models |

## Quick Start

### 1. Create a Model

```sql
-- Logistic regression for classification
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned'],
  auto_class_weights=TRUE
) AS
SELECT
  tenure,
  monthly_charges,
  total_charges,
  contract_type,
  churned
FROM `project.dataset.customer_data`
WHERE signup_date < '2024-01-01';  -- Training data
```

### 2. Evaluate the Model

```sql
SELECT * FROM ML.EVALUATE(
  MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.customer_data`
   WHERE signup_date >= '2024-01-01')  -- Test data
);
```

### 3. Make Predictions

```sql
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs
FROM ML.PREDICT(
  MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.new_customers`)
);
```

## Core ML Functions

| Function | Description |
|----------|-------------|
| `ML.EVALUATE` | Get model evaluation metrics |
| `ML.PREDICT` | Make predictions with trained model |
| `ML.EXPLAIN_PREDICT` | Predictions with feature attributions |
| `ML.FEATURE_INFO` | Get feature statistics |
| `ML.GLOBAL_EXPLAIN` | Global feature importance |
| `ML.CONFUSION_MATRIX` | Confusion matrix for classifiers |
| `ML.ROC_CURVE` | ROC curve data for binary classifiers |
| `ML.FORECAST` | Time series forecasting |
| `ML.DETECT_ANOMALIES` | Anomaly detection |
| `ML.RECOMMEND` | Generate recommendations |

## Model Training Examples

### Classification (Boosted Trees)

```sql
CREATE OR REPLACE MODEL `project.dataset.fraud_detector`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['is_fraud'],
  num_parallel_tree=5,
  max_iterations=50,
  learn_rate=0.1,
  early_stop=TRUE,
  data_split_method='AUTO_SPLIT'
) AS
SELECT
  transaction_amount,
  merchant_category,
  time_since_last_transaction,
  is_international,
  is_fraud
FROM `project.dataset.transactions`;
```

### Regression (Linear)

```sql
CREATE OR REPLACE MODEL `project.dataset.price_predictor`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['price'],
  optimize_strategy='BATCH_GRADIENT_DESCENT',
  l2_reg=0.1
) AS
SELECT
  square_feet,
  bedrooms,
  bathrooms,
  neighborhood,
  price
FROM `project.dataset.housing_data`;
```

### Time Series Forecasting

```sql
CREATE OR REPLACE MODEL `project.dataset.sales_forecast`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='daily_sales',
  auto_arima=TRUE,
  holiday_region='US',
  horizon=30
) AS
SELECT date, daily_sales
FROM `project.dataset.sales_history`
WHERE date < CURRENT_DATE();

-- Generate forecasts
SELECT * FROM ML.FORECAST(
  MODEL `project.dataset.sales_forecast`,
  STRUCT(30 AS horizon, 0.9 AS confidence_level)
);
```

### Clustering

```sql
CREATE OR REPLACE MODEL `project.dataset.customer_segments`
OPTIONS(
  model_type='KMEANS',
  num_clusters=5,
  kmeans_init_method='KMEANS++',
  standardize_features=TRUE
) AS
SELECT
  recency,
  frequency,
  monetary_value
FROM `project.dataset.customer_rfm`;

-- Assign clusters
SELECT
  customer_id,
  CENTROID_ID AS segment
FROM ML.PREDICT(
  MODEL `project.dataset.customer_segments`,
  (SELECT * FROM `project.dataset.customer_rfm`)
);
```

### Recommendations

```sql
CREATE OR REPLACE MODEL `project.dataset.product_recommender`
OPTIONS(
  model_type='MATRIX_FACTORIZATION',
  user_col='user_id',
  item_col='product_id',
  rating_col='rating',
  feedback_type='EXPLICIT',
  num_factors=20
) AS
SELECT user_id, product_id, rating
FROM `project.dataset.user_ratings`;

-- Generate recommendations
SELECT * FROM ML.RECOMMEND(
  MODEL `project.dataset.product_recommender`,
  (SELECT DISTINCT user_id FROM `project.dataset.active_users`),
  STRUCT(5 AS top_k)
);
```

## Model Evaluation

### Classification Metrics

```sql
-- Evaluation metrics
SELECT
  precision,
  recall,
  accuracy,
  f1_score,
  log_loss,
  roc_auc
FROM ML.EVALUATE(MODEL `project.dataset.classifier`);

-- Confusion matrix
SELECT * FROM ML.CONFUSION_MATRIX(
  MODEL `project.dataset.classifier`,
  (SELECT * FROM test_data)
);

-- ROC curve
SELECT * FROM ML.ROC_CURVE(
  MODEL `project.dataset.classifier`,
  (SELECT * FROM test_data)
);
```

### Regression Metrics

```sql
SELECT
  mean_absolute_error,
  mean_squared_error,
  mean_squared_log_error,
  median_absolute_error,
  r2_score,
  explained_variance
FROM ML.EVALUATE(MODEL `project.dataset.regressor`);
```

## Explainability

### Feature Importance

```sql
-- Global feature importance
SELECT *
FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.model`)
ORDER BY attribution DESC;
```

### Prediction Explanations

```sql
-- Per-prediction explanations
SELECT
  customer_id,
  predicted_label,
  top_feature_attributions
FROM ML.EXPLAIN_PREDICT(
  MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.customers` LIMIT 100),
  STRUCT(5 AS top_k_features)
);
```

## Model Management

### Get Model Information

```sql
-- Model metadata
SELECT * FROM ML.MODEL_INFO(MODEL `project.dataset.model`);

-- Training info
SELECT *
FROM ML.TRAINING_INFO(MODEL `project.dataset.model`);

-- Feature info
SELECT * FROM ML.FEATURE_INFO(MODEL `project.dataset.model`);
```

### Export Model

```sql
-- Export to Cloud Storage
EXPORT MODEL `project.dataset.model`
OPTIONS(URI='gs://bucket/model/');
```

### Drop Model

```sql
DROP MODEL IF EXISTS `project.dataset.model`;
```

## Hyperparameter Tuning

```sql
CREATE OR REPLACE MODEL `project.dataset.tuned_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  -- Hyperparameter search
  num_trials=20,
  max_parallel_trials=5,
  hparam_tuning_objectives=['ROC_AUC'],
  -- Parameter ranges
  learn_rate=HPARAM_RANGE(0.01, 0.3),
  max_tree_depth=HPARAM_CANDIDATES([4, 6, 8, 10]),
  subsample=HPARAM_RANGE(0.5, 1.0)
) AS
SELECT * FROM training_data;
```

## References

Load detailed documentation as needed:

- `MODEL_TYPES.md` - Complete list of model types with parameters
- `BEST_PRACTICES.md` - Tips for effective BQML usage
- `SQL_EXAMPLES.md` - Common SQL patterns and examples

## Scripts

Helper scripts for common operations:

- `validate_model.py` - Validate model configuration
- `export_metrics.py` - Export evaluation metrics to JSON

## Best Practices

1. **Data Splitting**: Use `data_split_method='AUTO_SPLIT'` for automatic train/test split
2. **Feature Engineering**: BQML handles basic transformations; pre-compute complex features
3. **Class Imbalance**: Use `auto_class_weights=TRUE` for imbalanced datasets
4. **Early Stopping**: Enable `early_stop=TRUE` to prevent overfitting
5. **Regularization**: Use L1/L2 regularization for linear models
6. **Model Selection**: Start simple (linear models) before complex (boosted trees, DNN)

## Limitations

- Training data must fit in BigQuery (no streaming)
- Limited deep learning capabilities vs. Vertex AI
- Some model types require specific data formats
- Hyperparameter tuning has limited search space
