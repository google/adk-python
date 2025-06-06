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

"""Data preprocessing tools for the Data Analysis Agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataPreprocessRequest(BaseModel):
    """Request to preprocess data."""
    
    data_id: str = Field(..., description="The identifier of the data to preprocess.")
    operation: str = Field(..., description="The preprocessing operation to perform.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the preprocessing operation."
    )


class DataPreprocessResponse(BaseModel):
    """Response from preprocessing data."""
    
    success: bool = Field(..., description="Whether the data was preprocessed successfully.")
    message: str = Field(..., description="A message describing the result.")
    data_id: Optional[str] = Field(
        None, description="An identifier for the preprocessed data."
    )
    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Statistics about the preprocessing operation."
    )


class DataPreprocessingTools:
    """Tools for preprocessing data."""

    def __init__(self, data_store: Dict[str, pd.DataFrame]):
        """Initialize the data preprocessing tools.
        
        Args:
            data_store: A dictionary mapping data IDs to DataFrames.
        """
        self._data_store = data_store

    async def preprocess_data(
        self, request: DataPreprocessRequest
    ) -> DataPreprocessResponse:
        """Preprocess data.
        
        Args:
            request: The request containing the data ID and preprocessing operation.
            
        Returns:
            A response indicating whether the data was preprocessed successfully.
        """
        try:
            # Check if the data ID exists
            if request.data_id not in self._data_store:
                return DataPreprocessResponse(
                    success=False,
                    message=f"Data ID '{request.data_id}' not found.",
                )
            
            # Get the data
            df = self._data_store[request.data_id].copy()
            
            # Perform the preprocessing
            operation_lower = request.operation.lower()
            parameters = request.parameters.copy()  # Create a copy to avoid modifying the original
            
            if operation_lower == "clean_missing_values":
                strategy = parameters.pop("strategy", "drop")
                df_processed = self._handle_missing_values(df, strategy, **parameters)
                statistics = {"original_shape": df.shape, "processed_shape": df_processed.shape}
            
            elif operation_lower == "handle_outliers":
                method = parameters.pop("method", "zscore")
                df_processed = self._handle_outliers(df, method, **parameters)
                statistics = {"original_shape": df.shape, "processed_shape": df_processed.shape}
            
            elif operation_lower == "engineer_features":
                operation = parameters.pop("operation", "polynomial")
                df_processed = self._engineer_features(df, operation, **parameters)
                statistics = {
                    "original_columns": len(df.columns),
                    "new_columns": len(df_processed.columns) - len(df.columns),
                    "processed_columns": len(df_processed.columns)
                }
            
            elif operation_lower == "encode_categorical":
                method = parameters.pop("method", "one_hot")
                df_processed = self._encode_categorical(df, method, **parameters)
                statistics = {
                    "original_columns": len(df.columns),
                    "processed_columns": len(df_processed.columns)
                }
            
            elif operation_lower == "normalize_data":
                method = parameters.pop("method", "minmax")
                df_processed = self._normalize_data(df, method, **parameters)
                statistics = {"original_shape": df.shape, "processed_shape": df_processed.shape}
            
            elif operation_lower == "remove_duplicates":
                df_processed = self._remove_duplicates(df, **parameters)
                statistics = {
                    "original_rows": len(df),
                    "duplicate_rows": len(df) - len(df_processed),
                    "processed_rows": len(df_processed)
                }
            
            elif operation_lower == "convert_types":
                df_processed = self._convert_types(df, **parameters)
                statistics = {"original_dtypes": df.dtypes.astype(str).to_dict(), 
                             "processed_dtypes": df_processed.dtypes.astype(str).to_dict()}
            
            else:
                return DataPreprocessResponse(
                    success=False,
                    message=f"Unsupported preprocessing operation '{request.operation}'.",
                )
            
            # Generate a unique ID for the preprocessed data
            data_id = f"preprocessed_{request.data_id}_{operation_lower}"
            self._data_store[data_id] = df_processed
            
            return DataPreprocessResponse(
                success=True,
                message=f"Successfully preprocessed data using '{request.operation}'.",
                data_id=data_id,
                statistics=statistics,
            )
        except Exception as e:
            logger.exception("Error preprocessing data")
            return DataPreprocessResponse(
                success=False,
                message=f"Error preprocessing data: {str(e)}",
            )

    def _handle_missing_values(
        self, df: pd.DataFrame, strategy: str, **kwargs
    ) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            df: The DataFrame to process.
            strategy: The strategy to use for handling missing values.
                Options: 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_constant'.
            **kwargs: Additional parameters for the strategy.
                
        Returns:
            The processed DataFrame.
        """
        if strategy == "drop":
            axis = kwargs.get("axis", 0)  # 0 for rows, 1 for columns
            how = kwargs.get("how", "any")  # 'any' or 'all'
            return df.dropna(axis=axis, how=how)
        
        elif strategy == "fill_mean":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
            return df
        
        elif strategy == "fill_median":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
            return df
        
        elif strategy == "fill_mode":
            columns = kwargs.get("columns", df.columns.tolist())
            for col in columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            return df
        
        elif strategy == "fill_constant":
            value = kwargs.get("value", 0)
            columns = kwargs.get("columns", df.columns.tolist())
            return df.fillna({col: value for col in columns})
        
        elif strategy == "fill_forward":
            columns = kwargs.get("columns", df.columns.tolist())
            for col in columns:
                df[col] = df[col].ffill()
            return df
        
        elif strategy == "fill_backward":
            columns = kwargs.get("columns", df.columns.tolist())
            for col in columns:
                df[col] = df[col].bfill()
            return df
        
        elif strategy == "fill_interpolate":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            method = kwargs.get("method", "linear")  # 'linear', 'time', 'index', 'values'
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].interpolate(method=method)
            return df
        
        else:
            raise ValueError(f"Unknown missing value handling strategy: {strategy}")

    def _handle_outliers(
        self, df: pd.DataFrame, method: str, **kwargs
    ) -> pd.DataFrame:
        """Detect and handle outliers in the data.
        
        Args:
            df: The DataFrame to process.
            method: The method to use for outlier detection.
                Options: 'zscore', 'iqr', 'percentile'.
            **kwargs: Additional parameters for the method.
                
        Returns:
            The processed DataFrame.
        """
        if method == "zscore":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            threshold = kwargs.get("threshold", 3)
            treatment = kwargs.get("treatment", "remove")
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Calculate mean and standard deviation, ignoring NaN values
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    
                    # Calculate z-scores
                    z_scores = np.abs((result_df[col] - mean) / std)
                    
                    # Identify outliers
                    outliers = z_scores > threshold
                    
                    if treatment == "remove":
                        # Remove rows with outliers
                        result_df = result_df[~outliers]
                    elif treatment == "cap":
                        # Cap outliers at threshold * std from mean
                        upper_bound = mean + threshold * std
                        lower_bound = mean - threshold * std
                        
                        # Apply capping
                        result_df.loc[result_df[col] > upper_bound, col] = upper_bound
                        result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                    elif treatment == "null":
                        # Replace outliers with NaN
                        result_df.loc[outliers, col] = np.nan
            
            return result_df
        
        elif method == "iqr":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            multiplier = kwargs.get("multiplier", 1.5)
            treatment = kwargs.get("treatment", "remove")
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - multiplier * iqr
                    upper_bound = q3 + multiplier * iqr
                    
                    outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                    
                    if treatment == "remove":
                        result_df = result_df[~outliers]
                    elif treatment == "cap":
                        result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                        result_df.loc[result_df[col] > upper_bound, col] = upper_bound
                    elif treatment == "null":
                        result_df.loc[outliers, col] = np.nan
            
            return result_df
        
        elif method == "percentile":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            lower_percentile = kwargs.get("lower_percentile", 0.01)
            upper_percentile = kwargs.get("upper_percentile", 0.99)
            treatment = kwargs.get("treatment", "cap")
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    lower_bound = result_df[col].quantile(lower_percentile)
                    upper_bound = result_df[col].quantile(upper_percentile)
                    
                    outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                    
                    if treatment == "remove":
                        result_df = result_df[~outliers]
                    elif treatment == "cap":
                        result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                        result_df.loc[result_df[col] > upper_bound, col] = upper_bound
                    elif treatment == "null":
                        result_df.loc[outliers, col] = np.nan
            
            return result_df
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _engineer_features(
        self, df: pd.DataFrame, operation: str, **kwargs
    ) -> pd.DataFrame:
        """Engineer new features from existing ones.
        
        Args:
            df: The DataFrame to process.
            operation: The feature engineering operation to perform.
                Options: 'polynomial', 'interaction', 'binning', 'date_features'.
            **kwargs: Additional parameters for the operation.
                
        Returns:
            The processed DataFrame with new features.
        """
        if operation == "polynomial":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            degree = kwargs.get("degree", 2)
            include_bias = kwargs.get("include_bias", False)
            
            try:
                from sklearn.preprocessing import PolynomialFeatures
                
                poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
                poly_features = poly.fit_transform(df[columns])
                
                feature_names = poly.get_feature_names_out(columns)
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                
                # Remove the original columns to avoid duplication
                poly_df = poly_df.drop(columns, axis=1)
                
                # Concatenate with the original DataFrame
                return pd.concat([df, poly_df], axis=1)
            except ImportError:
                logger.warning("sklearn is not installed. Using manual polynomial feature generation.")
                
                # Manual implementation for degree=2 (no higher degrees)
                if degree != 2:
                    raise ValueError("Manual polynomial feature generation only supports degree=2")
                
                result_df = df.copy()
                
                # Add squared terms
                for i, col1 in enumerate(columns):
                    result_df[f"{col1}^2"] = df[col1] ** 2
                    
                    # Add interaction terms
                    for j in range(i+1, len(columns)):
                        col2 = columns[j]
                        result_df[f"{col1}*{col2}"] = df[col1] * df[col2]
                
                return result_df
        
        elif operation == "interaction":
            columns = kwargs.get("columns", [])
            if len(columns) < 2:
                raise ValueError("At least two columns must be specified for interaction features")
            
            # Create interaction features
            result_df = df.copy()
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                        result_df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
            
            return result_df
        
        elif operation == "binning":
            column = kwargs.get("column")
            if not column:
                raise ValueError("Column must be specified for binning")
            
            bins = kwargs.get("bins", 10)
            labels = kwargs.get("labels", None)
            strategy = kwargs.get("strategy", "uniform")  # 'uniform', 'quantile', 'kmeans'
            
            result_df = df.copy()
            
            if strategy == "uniform":
                result_df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
            elif strategy == "quantile":
                result_df[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=labels)
            elif strategy == "kmeans":
                try:
                    from sklearn.cluster import KMeans
                    
                    kmeans = KMeans(n_clusters=bins)
                    result_df[f"{column}_binned"] = kmeans.fit_predict(df[[column]])
                except ImportError:
                    logger.warning("sklearn is not installed. Using quantile binning instead.")
                    result_df[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=labels)
            
            return result_df
        
        elif operation == "date_features":
            column = kwargs.get("column")
            if not column:
                raise ValueError("Column must be specified for date features")
            
            features = kwargs.get("features", ["year", "month", "day", "dayofweek", "quarter"])
            
            result_df = df.copy()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(result_df[column]):
                result_df[column] = pd.to_datetime(result_df[column], errors="coerce")
            
            # Extract date features
            if "year" in features:
                result_df[f"{column}_year"] = result_df[column].dt.year
            if "month" in features:
                result_df[f"{column}_month"] = result_df[column].dt.month
            if "day" in features:
                result_df[f"{column}_day"] = result_df[column].dt.day
            if "dayofweek" in features:
                result_df[f"{column}_dayofweek"] = result_df[column].dt.dayofweek
            if "quarter" in features:
                result_df[f"{column}_quarter"] = result_df[column].dt.quarter
            if "is_weekend" in features:
                result_df[f"{column}_is_weekend"] = result_df[column].dt.dayofweek >= 5
            if "is_month_end" in features:
                result_df[f"{column}_is_month_end"] = result_df[column].dt.is_month_end
            if "is_month_start" in features:
                result_df[f"{column}_is_month_start"] = result_df[column].dt.is_month_start
            if "is_quarter_end" in features:
                result_df[f"{column}_is_quarter_end"] = result_df[column].dt.is_quarter_end
            if "is_quarter_start" in features:
                result_df[f"{column}_is_quarter_start"] = result_df[column].dt.is_quarter_start
            if "is_year_end" in features:
                result_df[f"{column}_is_year_end"] = result_df[column].dt.is_year_end
            if "is_year_start" in features:
                result_df[f"{column}_is_year_start"] = result_df[column].dt.is_year_start
            
            return result_df
        
        elif operation == "text_features":
            column = kwargs.get("column")
            if not column:
                raise ValueError("Column must be specified for text features")
            
            features = kwargs.get("features", ["length", "word_count", "char_count"])
            
            result_df = df.copy()
            
            # Extract text features
            if "length" in features:
                result_df[f"{column}_length"] = result_df[column].str.len()
            if "word_count" in features:
                result_df[f"{column}_word_count"] = result_df[column].str.split().str.len()
            if "char_count" in features:
                result_df[f"{column}_char_count"] = result_df[column].str.replace(r'\s', '', regex=True).str.len()
            if "uppercase_count" in features:
                result_df[f"{column}_uppercase_count"] = result_df[column].str.count(r'[A-Z]')
            if "lowercase_count" in features:
                result_df[f"{column}_lowercase_count"] = result_df[column].str.count(r'[a-z]')
            if "digit_count" in features:
                result_df[f"{column}_digit_count"] = result_df[column].str.count(r'[0-9]')
            if "special_char_count" in features:
                result_df[f"{column}_special_char_count"] = result_df[column].str.count(r'[^a-zA-Z0-9\s]')
            
            return result_df
        
        else:
            raise ValueError(f"Unknown feature engineering operation: {operation}")

    def _encode_categorical(
        self, df: pd.DataFrame, method: str, **kwargs
    ) -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            df: The DataFrame to process.
            method: The encoding method to use.
                Options: 'one_hot', 'label', 'ordinal', 'target'.
            **kwargs: Additional parameters for the method.
                
        Returns:
            The processed DataFrame.
        """
        if method == "one_hot":
            columns = kwargs.get("columns", df.select_dtypes(include=["object", "category"]).columns.tolist())
            drop_first = kwargs.get("drop_first", False)
            
            result_df = df.copy()
            
            for col in columns:
                # Get dummies for the column
                dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=drop_first)
                
                # Concatenate with the result DataFrame
                result_df = pd.concat([result_df, dummies], axis=1)
                
                # Drop the original column if requested
                if kwargs.get("drop_original", True):
                    result_df = result_df.drop(col, axis=1)
            
            return result_df
        
        elif method == "label":
            columns = kwargs.get("columns", df.select_dtypes(include=["object", "category"]).columns.tolist())
            
            result_df = df.copy()
            
            for col in columns:
                # Create a mapping of unique values to integers
                unique_values = result_df[col].dropna().unique()
                mapping = {value: i for i, value in enumerate(unique_values)}
                
                # Apply the mapping
                result_df[f"{col}_encoded"] = result_df[col].map(mapping)
                
                # Drop the original column if requested
                if kwargs.get("drop_original", True):
                    result_df = result_df.drop(col, axis=1)
            
            return result_df
        
        elif method == "ordinal":
            columns = kwargs.get("columns", [])
            categories = kwargs.get("categories", {})
            
            if not columns:
                raise ValueError("Columns must be specified for ordinal encoding")
            
            result_df = df.copy()
            
            for col in columns:
                if col in categories:
                    # Create a mapping of categories to ordinal values
                    mapping = {category: i for i, category in enumerate(categories[col])}
                    
                    # Apply the mapping
                    result_df[f"{col}_encoded"] = result_df[col].map(mapping)
                    
                    # Drop the original column if requested
                    if kwargs.get("drop_original", True):
                        result_df = result_df.drop(col, axis=1)
                else:
                    raise ValueError(f"Categories must be specified for column '{col}'")
            
            return result_df
        
        elif method == "target":
            columns = kwargs.get("columns", [])
            target = kwargs.get("target")
            
            if not columns:
                raise ValueError("Columns must be specified for target encoding")
            
            if not target:
                raise ValueError("Target column must be specified for target encoding")
            
            result_df = df.copy()
            
            for col in columns:
                # Calculate the mean of the target for each category
                target_means = result_df.groupby(col)[target].mean()
                
                # Apply the mapping
                result_df[f"{col}_encoded"] = result_df[col].map(target_means)
                
                # Drop the original column if requested
                if kwargs.get("drop_original", True):
                    result_df = result_df.drop(col, axis=1)
            
            return result_df
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _normalize_data(
        self, df: pd.DataFrame, method: str, **kwargs
    ) -> pd.DataFrame:
        """Normalize or standardize numeric data.
        
        Args:
            df: The DataFrame to process.
            method: The normalization method to use.
                Options: 'minmax', 'zscore', 'robust', 'log', 'box-cox'.
            **kwargs: Additional parameters for the method.
                
        Returns:
            The processed DataFrame.
        """
        if method == "minmax":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            feature_range = kwargs.get("feature_range", (0, 1))
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    
                    if min_val == max_val:
                        # Handle constant columns
                        result_df[col] = feature_range[0]
                    else:
                        # Min-max scaling
                        result_df[col] = (result_df[col] - min_val) / (max_val - min_val)
                        
                        # Scale to feature range
                        result_df[col] = result_df[col] * (feature_range[1] - feature_range[0]) + feature_range[0]
            
            return result_df
        
        elif method == "zscore":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    
                    if std == 0:
                        # Handle constant columns
                        result_df[col] = 0
                    else:
                        # Z-score standardization
                        result_df[col] = (result_df[col] - mean) / std
            
            return result_df
        
        elif method == "robust":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    median = result_df[col].median()
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr == 0:
                        # Handle columns with zero IQR
                        result_df[col] = 0
                    else:
                        # Robust scaling
                        result_df[col] = (result_df[col] - median) / iqr
            
            return result_df
        
        elif method == "log":
            columns = kwargs.get("columns", df.select_dtypes(include=["number"]).columns.tolist())
            base = kwargs.get("base", np.e)
            
            result_df = df.copy()
            
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Handle negative and zero values
                    min_val = result_df[col].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        result_df[col] = result_df[col] + shift
                    
                    # Apply log transformation
                    if base == np.e:
                        result_df[col] = np.log(result_df[col])
                    else:
                        result_df[col] = np.log(result_df[col]) / np.log(base)
            
            return result_df
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _remove_duplicates(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame.
        
        Args:
            df: The DataFrame to process.
            **kwargs: Additional parameters for duplicate removal.
                
        Returns:
            The processed DataFrame.
        """
        subset = kwargs.get("subset", None)  # Columns to consider for identifying duplicates
        keep = kwargs.get("keep", "first")  # 'first', 'last', or False
        
        return df.drop_duplicates(subset=subset, keep=keep)

    def _convert_types(
        self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Convert column data types.
        
        Args:
            df: The DataFrame to process.
            **kwargs: Additional parameters for type conversion.
                
        Returns:
            The processed DataFrame.
        """
        type_mappings = kwargs.get("type_mappings", {})
        
        if not type_mappings:
            raise ValueError("Type mappings must be specified for type conversion")
        
        result_df = df.copy()
        
        for col, dtype in type_mappings.items():
            if col in result_df.columns:
                try:
                    if dtype == "datetime":
                        result_df[col] = pd.to_datetime(result_df[col], errors="coerce")
                    elif dtype == "category":
                        result_df[col] = result_df[col].astype("category")
                    else:
                        result_df[col] = result_df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Error converting column '{col}' to type '{dtype}': {str(e)}")
            else:
                logger.warning(f"Column '{col}' not found in DataFrame")
        
        return result_df

