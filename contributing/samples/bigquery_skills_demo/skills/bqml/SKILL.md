---
name: bqml
description: BigQuery ML - Train, evaluate, and deploy machine learning models using SQL. Supports regression, classification, clustering, time series forecasting, and deep learning.
---

# BQML Skill (BigQuery Machine Learning)

Train, evaluate, and deploy ML models directly in BigQuery using SQL.

## Supported Model Types

| Model Type | Use Case | SQL Option |
|------------|----------|------------|
| LINEAR_REG | Numeric prediction | `model_type='LINEAR_REG'` |
| LOGISTIC_REG | Binary/multiclass classification | `model_type='LOGISTIC_REG'` |
| KMEANS | Customer segmentation, clustering | `model_type='KMEANS'` |
| BOOSTED_TREE_REGRESSOR | Numeric prediction with XGBoost | `model_type='BOOSTED_TREE_REGRESSOR'` |
| BOOSTED_TREE_CLASSIFIER | Classification with XGBoost | `model_type='BOOSTED_TREE_CLASSIFIER'` |
| RANDOM_FOREST_REGRESSOR | Ensemble numeric prediction | `model_type='RANDOM_FOREST_REGRESSOR'` |
| RANDOM_FOREST_CLASSIFIER | Ensemble classification | `model_type='RANDOM_FOREST_CLASSIFIER'` |
| ARIMA_PLUS | Time series forecasting | `model_type='ARIMA_PLUS'` |
| DNN_REGRESSOR | Deep learning regression | `model_type='DNN_REGRESSOR'` |
| DNN_CLASSIFIER | Deep learning classification | `model_type='DNN_CLASSIFIER'` |

## Core Workflow

### Step 1: Train a Model
```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
    model_type='LINEAR_REG',
    input_label_cols=['target_column'],
    enable_global_explain=TRUE
) AS
SELECT feature1, feature2, feature3, target_column
FROM `project.dataset.training_data`
WHERE target_column IS NOT NULL;
```

### Step 2: Evaluate the Model
```sql
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.model_name`);
```

### Step 3: Get Feature Importance
```sql
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.model_name`);
```

### Step 4: Make Predictions
```sql
SELECT * FROM ML.PREDICT(
    MODEL `project.dataset.model_name`,
    (SELECT feature1, feature2, feature3 FROM `project.dataset.new_data`)
);
```

### Step 5: Explain Predictions
```sql
SELECT * FROM ML.EXPLAIN_PREDICT(
    MODEL `project.dataset.model_name`,
    (SELECT feature1, feature2, feature3 FROM `project.dataset.new_data`),
    STRUCT(3 as top_k_features)
);
```

## Example: Penguin Body Mass Prediction

```sql
-- Train model
CREATE OR REPLACE MODEL `project.bqml_demo.penguin_weight`
OPTIONS(
    model_type='LINEAR_REG',
    input_label_cols=['body_mass_g'],
    enable_global_explain=TRUE
) AS
SELECT species, island, culmen_length_mm, culmen_depth_mm,
       flipper_length_mm, sex, body_mass_g
FROM `bigquery-public-data.ml_datasets.penguins`
WHERE body_mass_g IS NOT NULL AND sex IS NOT NULL;

-- Evaluate
SELECT * FROM ML.EVALUATE(MODEL `project.bqml_demo.penguin_weight`);

-- Feature importance
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `project.bqml_demo.penguin_weight`);

-- Predict
SELECT predicted_body_mass_g, species, island
FROM ML.PREDICT(
    MODEL `project.bqml_demo.penguin_weight`,
    (SELECT 'Adelie' as species, 'Torgersen' as island,
            39.1 as culmen_length_mm, 18.7 as culmen_depth_mm,
            181.0 as flipper_length_mm, 'MALE' as sex)
);
```

## Example: K-Means Clustering

```sql
-- Create clustering model
CREATE OR REPLACE MODEL `project.bqml_demo.penguin_clusters`
OPTIONS(
    model_type='KMEANS',
    num_clusters=3,
    standardize_features=TRUE
) AS
SELECT culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
FROM `bigquery-public-data.ml_datasets.penguins`
WHERE body_mass_g IS NOT NULL;

-- Get cluster assignments
SELECT * FROM ML.PREDICT(
    MODEL `project.bqml_demo.penguin_clusters`,
    (SELECT culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
     FROM `bigquery-public-data.ml_datasets.penguins`
     WHERE body_mass_g IS NOT NULL)
);

-- Analyze cluster centroids
SELECT * FROM ML.CENTROIDS(MODEL `project.bqml_demo.penguin_clusters`);
```

## Example: XGBoost Classification

```sql
CREATE OR REPLACE MODEL `project.bqml_demo.species_classifier`
OPTIONS(
    model_type='BOOSTED_TREE_CLASSIFIER',
    input_label_cols=['species'],
    num_parallel_tree=1,
    max_tree_depth=6,
    subsample=0.8,
    data_split_method='AUTO_SPLIT'
) AS
SELECT island, culmen_length_mm, culmen_depth_mm,
       flipper_length_mm, body_mass_g, sex, species
FROM `bigquery-public-data.ml_datasets.penguins`
WHERE species IS NOT NULL AND sex IS NOT NULL;

-- Confusion matrix
SELECT * FROM ML.CONFUSION_MATRIX(MODEL `project.bqml_demo.species_classifier`);

-- ROC curve (for binary classification)
SELECT * FROM ML.ROC_CURVE(MODEL `project.bqml_demo.species_classifier`);
```

## Key ML Functions

- `ML.EVALUATE()` - Model performance metrics
- `ML.PREDICT()` - Generate predictions
- `ML.EXPLAIN_PREDICT()` - Predictions with feature attributions
- `ML.GLOBAL_EXPLAIN()` - Overall feature importance
- `ML.FEATURE_IMPORTANCE()` - Feature weights for tree models
- `ML.CONFUSION_MATRIX()` - Classification matrix
- `ML.ROC_CURVE()` - ROC curve data
- `ML.CENTROIDS()` - K-means cluster centers
- `ML.TRAINING_INFO()` - Training run details
