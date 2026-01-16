# BQML Model Types Reference

## Supervised Learning

### Linear Regression
- **Type**: `LINEAR_REG`
- **Use case**: Predict continuous numeric values
- **Example**: Predict sales, prices, quantities

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['target_column']
) AS
SELECT * FROM `project.dataset.training_data`
```

### Logistic Regression
- **Type**: `LOGISTIC_REG`
- **Use case**: Binary or multiclass classification
- **Example**: Predict churn, fraud detection

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label_column'],
  auto_class_weights=TRUE
) AS
SELECT * FROM `project.dataset.training_data`
```

### Boosted Tree Classifier
- **Type**: `BOOSTED_TREE_CLASSIFIER`
- **Use case**: Complex classification with feature interactions
- **Parameters**: `num_parallel_tree`, `max_iterations`, `learn_rate`

### Boosted Tree Regressor
- **Type**: `BOOSTED_TREE_REGRESSOR`
- **Use case**: Complex regression with non-linear relationships

### Random Forest Classifier
- **Type**: `RANDOM_FOREST_CLASSIFIER`
- **Use case**: Ensemble classification

### Random Forest Regressor
- **Type**: `RANDOM_FOREST_REGRESSOR`
- **Use case**: Ensemble regression

### DNN Classifier
- **Type**: `DNN_CLASSIFIER`
- **Use case**: Deep learning for classification
- **Parameters**: `hidden_units`, `dropout`, `batch_size`

### DNN Regressor
- **Type**: `DNN_REGRESSOR`
- **Use case**: Deep learning for regression

### Wide & Deep
- **Type**: `DNN_LINEAR_COMBINED_CLASSIFIER` / `DNN_LINEAR_COMBINED_REGRESSOR`
- **Use case**: Combines memorization (wide) with generalization (deep)

## Unsupervised Learning

### K-Means Clustering
- **Type**: `KMEANS`
- **Use case**: Customer segmentation, anomaly detection
- **Parameters**: `num_clusters`, `kmeans_init_method`

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='KMEANS',
  num_clusters=5
) AS
SELECT feature1, feature2 FROM `project.dataset.data`
```

### PCA (Principal Component Analysis)
- **Type**: `PCA`
- **Use case**: Dimensionality reduction
- **Parameters**: `num_principal_components`

### Autoencoder
- **Type**: `AUTOENCODER`
- **Use case**: Anomaly detection, feature learning

## Time Series

### ARIMA Plus
- **Type**: `ARIMA_PLUS`
- **Use case**: Time series forecasting
- **Parameters**: `time_series_timestamp_col`, `time_series_data_col`, `horizon`

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='sales',
  horizon=30
) AS
SELECT date, sales FROM `project.dataset.time_series_data`
```

## Matrix Factorization

### Matrix Factorization
- **Type**: `MATRIX_FACTORIZATION`
- **Use case**: Recommendation systems
- **Parameters**: `user_col`, `item_col`, `rating_col`, `num_factors`

## Imported Models

### TensorFlow
- **Type**: `TENSORFLOW`
- **Use case**: Import trained TensorFlow SavedModel

### ONNX
- **Type**: `ONNX`
- **Use case**: Import ONNX models

### XGBoost
- **Type**: `XGBOOST`
- **Use case**: Import XGBoost models

## LLM Integration

### Remote Models
- **Type**: `remote` with `REMOTE_SERVICE_TYPE`
- **Use case**: Connect to Vertex AI LLMs
- **Supported**: Gemini, PaLM, Claude (via Model Garden)

```sql
CREATE OR REPLACE MODEL `project.dataset.llm_model`
REMOTE WITH CONNECTION `project.region.connection_name`
OPTIONS(
  REMOTE_SERVICE_TYPE='CLOUD_AI_LARGE_LANGUAGE_MODEL_V1',
  endpoint='gemini-pro'
)
```
