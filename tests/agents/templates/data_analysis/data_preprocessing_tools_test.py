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

"""Tests for data preprocessing tools."""

import asyncio
import pandas as pd
import pytest

from google.adk.agents.templates.data_analysis.data_preprocessing_tools import (
    DataPreprocessingTools,
    DataPreprocessRequest,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "numeric_col": [1, 2, None, 4, 5, 100],  # Contains missing value and outlier
        "categorical_col": ["A", "B", "A", None, "C", "B"],  # Contains missing value
        "date_col": pd.date_range(start="2023-01-01", periods=6, freq="D"),
    })


@pytest.fixture
def preprocessing_tools(sample_dataframe):
    """Create a DataPreprocessingTools instance with the sample DataFrame."""
    data_store = {"test_data": sample_dataframe}
    return DataPreprocessingTools(data_store)


@pytest.mark.asyncio
async def test_handle_missing_values(preprocessing_tools):
    """Test handling missing values."""
    # Test drop strategy
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="clean_missing_values",
        parameters={"strategy": "drop"}
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    assert preprocessing_tools._data_store[response.data_id].shape[0] == 4  # 2 rows with NaN dropped
    
    # Test fill_mean strategy
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="clean_missing_values",
        parameters={"strategy": "fill_mean"}
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    assert not pd.isna(df["numeric_col"]).any()  # No missing values in numeric column
    assert pd.isna(df["categorical_col"]).any()  # Still missing values in categorical column


@pytest.mark.asyncio
async def test_handle_outliers(preprocessing_tools):
    """Test handling outliers."""
    # Test zscore method with a lower threshold to ensure capping
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="handle_outliers",
        parameters={
            "method": "zscore",
            "columns": ["numeric_col"],
            "threshold": 1.5,  # Lower threshold to ensure 100 is an outlier
            "treatment": "cap"
        }
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    
    # Calculate the expected upper bound
    mean = preprocessing_tools._data_store["test_data"]["numeric_col"].mean()
    std = preprocessing_tools._data_store["test_data"]["numeric_col"].std()
    upper_bound = mean + 1.5 * std
    
    # Check if the outlier was capped at the upper bound
    assert df["numeric_col"].max() <= upper_bound


@pytest.mark.asyncio
async def test_engineer_features(preprocessing_tools):
    """Test feature engineering."""
    # Test date features
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="engineer_features",
        parameters={
            "operation": "date_features",
            "column": "date_col",
            "features": ["year", "month", "day", "dayofweek"]
        }
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    assert "date_col_year" in df.columns
    assert "date_col_month" in df.columns
    assert "date_col_day" in df.columns
    assert "date_col_dayofweek" in df.columns


@pytest.mark.asyncio
async def test_encode_categorical(preprocessing_tools):
    """Test categorical encoding."""
    # Test one-hot encoding
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="encode_categorical",
        parameters={
            "method": "one_hot",
            "columns": ["categorical_col"],
            "drop_first": False
        }
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    assert "categorical_col_A" in df.columns
    assert "categorical_col_B" in df.columns
    assert "categorical_col_C" in df.columns


@pytest.mark.asyncio
async def test_normalize_data(preprocessing_tools):
    """Test data normalization."""
    # Test minmax normalization
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="normalize_data",
        parameters={
            "method": "minmax",
            "columns": ["numeric_col"],
            "feature_range": (0, 1)
        }
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    # Check if values are between 0 and 1
    numeric_values = df["numeric_col"].dropna()
    assert numeric_values.min() >= 0
    assert numeric_values.max() <= 1


@pytest.mark.asyncio
async def test_remove_duplicates(preprocessing_tools):
    """Test removing duplicates."""
    # Add duplicate rows to the test data
    df = preprocessing_tools._data_store["test_data"].copy()
    df = pd.concat([df, df.iloc[[0, 1]]])
    preprocessing_tools._data_store["test_data_with_duplicates"] = df
    
    request = DataPreprocessRequest(
        data_id="test_data_with_duplicates",
        operation="remove_duplicates",
        parameters={}
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    result_df = preprocessing_tools._data_store[response.data_id]
    assert len(result_df) == len(df) - 2  # 2 duplicate rows should be removed


@pytest.mark.asyncio
async def test_convert_types(preprocessing_tools):
    """Test converting data types."""
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="convert_types",
        parameters={
            "type_mappings": {
                "numeric_col": "float32",
                "categorical_col": "category"
            }
        }
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert response.success
    assert response.data_id is not None
    df = preprocessing_tools._data_store[response.data_id]
    assert df["numeric_col"].dtype == "float32"
    assert df["categorical_col"].dtype.name == "category"


@pytest.mark.asyncio
async def test_invalid_data_id(preprocessing_tools):
    """Test behavior with invalid data ID."""
    request = DataPreprocessRequest(
        data_id="nonexistent_data",
        operation="clean_missing_values",
        parameters={"strategy": "drop"}
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert not response.success
    assert "not found" in response.message


@pytest.mark.asyncio
async def test_invalid_operation(preprocessing_tools):
    """Test behavior with invalid operation."""
    request = DataPreprocessRequest(
        data_id="test_data",
        operation="invalid_operation",
        parameters={}
    )
    response = await preprocessing_tools.preprocess_data(request)
    
    assert not response.success
    assert "Unsupported" in response.message

