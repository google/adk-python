# BQML SQL Examples

## Creating Models

### Classification Model
```sql
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  auto_class_weights=TRUE,
  max_iterations=50,
  early_stop=TRUE
) AS
SELECT
  customer_id,
  tenure_months,
  monthly_charges,
  total_charges,
  contract_type,
  payment_method,
  churned
FROM `project.dataset.customer_data`
WHERE data_split = 'TRAIN'
```

### Regression Model
```sql
CREATE OR REPLACE MODEL `project.dataset.sales_forecast`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['sales'],
  l2_reg=0.1,
  max_iterations=100
) AS
SELECT
  product_category,
  region,
  month,
  marketing_spend,
  sales
FROM `project.dataset.sales_data`
```

### Time Series Forecasting
```sql
CREATE OR REPLACE MODEL `project.dataset.demand_forecast`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='demand',
  auto_arima=TRUE,
  horizon=30,
  holiday_region='US'
) AS
SELECT
  date,
  demand
FROM `project.dataset.daily_demand`
```

### Clustering
```sql
CREATE OR REPLACE MODEL `project.dataset.customer_segments`
OPTIONS(
  model_type='KMEANS',
  num_clusters=5,
  standardize_features=TRUE
) AS
SELECT
  recency,
  frequency,
  monetary_value
FROM `project.dataset.rfm_data`
```

## Evaluating Models

### Get Evaluation Metrics
```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.customer_data` WHERE data_split = 'TEST')
)
```

### Confusion Matrix
```sql
SELECT *
FROM ML.CONFUSION_MATRIX(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.customer_data` WHERE data_split = 'TEST')
)
```

### ROC Curve
```sql
SELECT *
FROM ML.ROC_CURVE(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.customer_data` WHERE data_split = 'TEST')
)
```

### Feature Importance
```sql
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.churn_model`)
ORDER BY importance_weight DESC
```

## Making Predictions

### Basic Prediction
```sql
SELECT *
FROM ML.PREDICT(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.new_customers`)
)
```

### Prediction with Threshold
```sql
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability
FROM ML.PREDICT(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.new_customers`)
)
WHERE predicted_churned_probs[OFFSET(1)].prob > 0.7
```

### Explainable Predictions
```sql
SELECT *
FROM ML.EXPLAIN_PREDICT(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.new_customers`),
  STRUCT(3 AS top_k_features)
)
```

### Time Series Forecast
```sql
SELECT *
FROM ML.FORECAST(MODEL `project.dataset.demand_forecast`,
  STRUCT(30 AS horizon, 0.95 AS confidence_level)
)
```

## Advanced Patterns

### Batch Scoring with Partitioning
```sql
CREATE OR REPLACE TABLE `project.dataset.predictions`
PARTITION BY DATE(prediction_date)
AS
SELECT
  CURRENT_DATE() AS prediction_date,
  p.*
FROM ML.PREDICT(MODEL `project.dataset.model`,
  (SELECT * FROM `project.dataset.input_data`)
) p
```

### Model Comparison
```sql
WITH model_metrics AS (
  SELECT 'model_v1' AS model, * FROM ML.EVALUATE(MODEL `project.dataset.model_v1`)
  UNION ALL
  SELECT 'model_v2' AS model, * FROM ML.EVALUATE(MODEL `project.dataset.model_v2`)
)
SELECT model, precision, recall, f1_score, roc_auc
FROM model_metrics
```

### Incremental Training
```sql
CREATE OR REPLACE MODEL `project.dataset.model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  warm_start=TRUE  -- Continue from existing model
) AS
SELECT * FROM `project.dataset.new_training_data`
```

### Transform at Prediction Time
```sql
SELECT *
FROM ML.PREDICT(MODEL `project.dataset.model`,
  (
    SELECT
      IFNULL(feature1, 0) AS feature1,
      LOG(feature2 + 1) AS feature2,
      LOWER(category) AS category
    FROM `project.dataset.raw_input`
  )
)
```
