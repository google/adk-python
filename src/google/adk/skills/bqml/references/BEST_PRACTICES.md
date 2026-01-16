# BQML Best Practices

## Data Preparation

### Feature Engineering
- **Normalize numeric features**: Use `ML.STANDARD_SCALER` or `ML.MIN_MAX_SCALER`
- **Handle missing values**: BQML handles NULLs automatically, but explicit imputation may improve results
- **Encode categoricals**: BQML auto-encodes, but one-hot encoding can help for high-cardinality features
- **Create interaction features**: Combine related features when domain knowledge suggests

### Data Quality
- Remove duplicates before training
- Handle outliers appropriately (cap, remove, or transform)
- Ensure sufficient training data (rule of thumb: 10x features minimum)
- Check for data leakage from target to features

### Train/Test Split
```sql
-- Use a hash-based split for reproducibility
SELECT *,
  MOD(ABS(FARM_FINGERPRINT(CAST(id AS STRING))), 10) AS split_group
FROM table
-- split_group < 8 for training, >= 8 for evaluation
```

## Model Selection

### Choose the Right Model Type
| Problem Type | Recommended Models |
|-------------|-------------------|
| Binary classification | LOGISTIC_REG, BOOSTED_TREE_CLASSIFIER |
| Multi-class | LOGISTIC_REG (with auto_class_weights), DNN_CLASSIFIER |
| Regression | LINEAR_REG, BOOSTED_TREE_REGRESSOR |
| Time series | ARIMA_PLUS |
| Clustering | KMEANS |
| Recommendations | MATRIX_FACTORIZATION |

### Start Simple
1. Begin with linear models (fast, interpretable)
2. Move to boosted trees if linear underperforms
3. Use deep learning only when data volume justifies complexity

## Hyperparameter Tuning

### Automated Tuning
```sql
CREATE OR REPLACE MODEL `project.dataset.model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  num_trials=20,
  max_parallel_trials=5,
  hparam_tuning_objectives=['ROC_AUC']
) AS
SELECT * FROM training_data
```

### Key Parameters by Model Type

**Boosted Trees**:
- `num_parallel_tree`: 1-10 (start with 1)
- `max_iterations`: 20-500
- `learn_rate`: 0.01-0.3
- `subsample`: 0.5-1.0

**DNN**:
- `hidden_units`: Start with [64, 32] or [128, 64, 32]
- `dropout`: 0.1-0.5
- `batch_size`: 256-4096

## Evaluation

### Classification Metrics
- **Precision/Recall**: When classes are imbalanced
- **ROC AUC**: Overall discriminative ability
- **Log Loss**: For probability calibration
- **Confusion Matrix**: Understand error types

### Regression Metrics
- **RMSE**: Penalizes large errors
- **MAE**: More robust to outliers
- **R-squared**: Explained variance
- **MAPE**: Percentage error interpretation

### Cross-Validation
```sql
-- Use k-fold cross-validation
SELECT *
FROM ML.CROSS_VALIDATE(
  MODEL `project.dataset.model`,
  TABLE `project.dataset.data`,
  STRUCT(5 AS num_folds)
)
```

## Production Deployment

### Model Versioning
- Use descriptive model names with versions: `model_v1`, `model_v2`
- Document model changes in metadata
- Keep training queries in version control

### Monitoring
- Track prediction drift over time
- Monitor feature distributions
- Set up alerts for model performance degradation
- Retrain on schedule or when metrics decline

### Cost Optimization
- Use `DATA_SPLIT_METHOD='RANDOM'` for large datasets
- Limit `max_iterations` during experimentation
- Use slots reservation for production training
- Consider batch predictions vs. real-time for cost

## Common Pitfalls

1. **Data Leakage**: Features that contain target information
2. **Class Imbalance**: Use `auto_class_weights=TRUE` or resampling
3. **Overfitting**: Monitor train vs. eval metrics, use regularization
4. **Feature Scaling**: Required for some models (DNN, linear with regularization)
5. **Timestamp Handling**: Ensure proper time-based splits for time series
