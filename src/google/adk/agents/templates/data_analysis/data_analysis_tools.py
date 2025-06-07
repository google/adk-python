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

"""Data Analysis tools for the Data Analysis Agent."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.tools.base_tool import BaseTool

from .data_preprocessing_tools import DataPreprocessRequest, DataPreprocessResponse, DataPreprocessingTools
from .data_source_tools import (
    DataSourceTools, 
    FileUploadRequest, FileUploadResponse,
    GoogleSheetRequest, GoogleSheetResponse,
    DatabaseConnectionRequest, DatabaseConnectionResponse,
    APIDataRequest, APIDataResponse
)
from .analysis_detector import (
    AnalysisDetector,
    DetectAnalysisTypeRequest, DetectAnalysisTypeResponse
)

logger = logging.getLogger(__name__)


class LoadDataRequest(BaseModel):
  """Request to load data from a source."""
  
  source: str = Field(..., description="The source of the data to load.")
  format: str = Field(
      "auto", description="The format of the data (csv, json, excel, etc.)."
  )
  options: Dict[str, Any] = Field(
      default_factory=dict, description="Additional options for loading the data."
  )


class LoadDataResponse(BaseModel):
  """Response from loading data."""
  
  success: bool = Field(..., description="Whether the data was loaded successfully.")
  message: str = Field(..., description="A message describing the result.")
  data_id: Optional[str] = Field(
      None, description="An identifier for the loaded data."
  )


class TransformDataRequest(BaseModel):
  """Request to transform data."""
  
  data_id: str = Field(..., description="The identifier of the data to transform.")
  operation: str = Field(..., description="The transformation operation to perform.")
  parameters: Dict[str, Any] = Field(
      default_factory=dict, description="Parameters for the transformation."
  )


class TransformDataResponse(BaseModel):
  """Response from transforming data."""
  
  success: bool = Field(..., description="Whether the data was transformed successfully.")
  message: str = Field(..., description="A message describing the result.")
  data_id: Optional[str] = Field(
      None, description="An identifier for the transformed data."
  )


class AnalyzeDataRequest(BaseModel):
  """Request to analyze data."""
  
  data_id: str = Field(..., description="The identifier of the data to analyze.")
  analysis_type: str = Field(..., description="The type of analysis to perform.")
  parameters: Dict[str, Any] = Field(
      default_factory=dict, description="Parameters for the analysis."
  )


class AnalyzeDataResponse(BaseModel):
  """Response from analyzing data."""
  
  success: bool = Field(..., description="Whether the analysis was successful.")
  message: str = Field(..., description="A message describing the result.")
  results: Dict[str, Any] = Field(
      default_factory=dict, description="The results of the analysis."
  )


class VisualizeDataRequest(BaseModel):
  """Request to visualize data."""
  
  data_id: str = Field(..., description="The identifier of the data to visualize.")
  visualization_type: str = Field(..., description="The type of visualization to create.")
  parameters: Dict[str, Any] = Field(
      default_factory=dict, description="Parameters for the visualization."
  )


class VisualizeDataResponse(BaseModel):
  """Response from visualizing data."""
  
  success: bool = Field(..., description="Whether the visualization was successful.")
  message: str = Field(..., description="A message describing the result.")
  visualization_path: Optional[str] = Field(
      None, description="The path to the visualization file."
  )


class DataAnalysisToolset(BaseTool):
  """A toolset for data analysis tasks."""
  
  name: str = "data_analysis_toolset"
  description: str = "A set of tools for data analysis tasks."
  
  def __init__(
      self,
      data_sources: Optional[List[str]] = None,
      analysis_types: Optional[List[str]] = None,
      visualization_types: Optional[List[str]] = None,
      memory_service: Optional[BaseMemoryService] = None,
  ):
    """Initialize the DataAnalysisToolset.
    
    Args:
      data_sources: List of data sources to analyze.
      analysis_types: List of analysis types to perform.
      visualization_types: List of visualization types to generate.
      memory_service: Optional memory service for persisting analysis results.
    """
    super().__init__()
    self.data_sources = data_sources or []
    self.analysis_types = analysis_types or []
    self.visualization_types = visualization_types or []
    self.memory_service = memory_service
    self._data_store: Dict[str, pd.DataFrame] = {}
    self._temp_dir = tempfile.mkdtemp()
    self._preprocessing_tools = DataPreprocessingTools(self._data_store)
    self._data_source_tools = DataSourceTools(self._data_store, self._temp_dir)
    self._analysis_detector = AnalysisDetector(self._data_store)
  
  async def load_data(self, request: LoadDataRequest) -> LoadDataResponse:
    """Load data from a source.
    
    Args:
      request: The request containing the source and format information.
      
    Returns:
      A response indicating whether the data was loaded successfully.
    """
    try:
      # Check if the source is in the list of allowed data sources
      if self.data_sources and request.source not in self.data_sources:
        return LoadDataResponse(
            success=False,
            message=f"Data source '{request.source}' is not in the list of allowed sources.",
        )
      
      # Determine the format if set to auto
      format_lower = request.format.lower()
      if format_lower == "auto":
        if request.source.endswith(".csv"):
          format_lower = "csv"
        elif request.source.endswith(".json"):
          format_lower = "json"
        elif request.source.endswith((".xls", ".xlsx")):
          format_lower = "excel"
        else:
          return LoadDataResponse(
              success=False,
              message=f"Could not determine format for source '{request.source}'.",
          )
      
      # Load the data based on the format
      if format_lower == "csv":
        df = pd.read_csv(request.source, **request.options)
      elif format_lower == "json":
        df = pd.read_json(request.source, **request.options)
      elif format_lower == "excel":
        df = pd.read_excel(request.source, **request.options)
      else:
        return LoadDataResponse(
            success=False,
            message=f"Unsupported format '{format_lower}'.",
        )
      
      # Generate a unique ID for the data
      data_id = f"data_{len(self._data_store) + 1}"
      self._data_store[data_id] = df
      
      return LoadDataResponse(
          success=True,
          message=f"Successfully loaded data from '{request.source}'.",
          data_id=data_id,
      )
    except Exception as e:
      logger.exception("Error loading data")
      return LoadDataResponse(
          success=False,
          message=f"Error loading data: {str(e)}",
      )
  
  async def upload_file(self, request: FileUploadRequest) -> FileUploadResponse:
    """Upload a file for data analysis.
    
    Args:
      request: The request containing the file path and type.
      
    Returns:
      A response indicating whether the file was uploaded successfully.
    """
    return await self._data_source_tools.upload_file(request)
  
  async def load_google_sheet(self, request: GoogleSheetRequest) -> GoogleSheetResponse:
    """Load data from a Google Sheet.
    
    Args:
      request: The request containing the Google Sheet URL and options.
      
    Returns:
      A response indicating whether the sheet was loaded successfully.
    """
    return await self._data_source_tools.load_google_sheet(request)
  
  async def connect_to_database(self, request: DatabaseConnectionRequest) -> DatabaseConnectionResponse:
    """Connect to a database and load data.
    
    Args:
      request: The request containing the database connection information.
      
    Returns:
      A response indicating whether the connection was successful.
    """
    return await self._data_source_tools.connect_to_database(request)
  
  async def load_api_data(self, request: APIDataRequest) -> APIDataResponse:
    """Load data from an API.
    
    Args:
      request: The request containing the API URL and options.
      
    Returns:
      A response indicating whether the data was loaded successfully.
    """
    return await self._data_source_tools.load_api_data(request)
  
  async def detect_analysis_type(self, request: DetectAnalysisTypeRequest) -> DetectAnalysisTypeResponse:
    """Detect analysis types for a dataset.
    
    Args:
      request: The request containing the data ID.
      
    Returns:
      A response containing the detected analysis types and objectives.
    """
    return await self._analysis_detector.detect_analysis_type(request)
  
  async def transform_data(
      self, request: TransformDataRequest
  ) -> TransformDataResponse:
    """Transform data.
    
    Args:
      request: The request containing the data ID and transformation operation.
      
    Returns:
      A response indicating whether the data was transformed successfully.
    """
    try:
      # Check if the data ID exists
      if request.data_id not in self._data_store:
        return TransformDataResponse(
            success=False,
            message=f"Data ID '{request.data_id}' not found.",
        )
      
      # Get the data
      df = self._data_store[request.data_id]
      
      # Perform the transformation
      operation_lower = request.operation.lower()
      if operation_lower == "filter":
        # Filter the data based on a condition
        if "condition" not in request.parameters:
          return TransformDataResponse(
              success=False,
              message="Missing 'condition' parameter for filter operation.",
          )
        
        # Use pandas query to filter the data
        df_transformed = df.query(request.parameters["condition"])
      
      elif operation_lower == "select":
        # Select specific columns
        if "columns" not in request.parameters:
          return TransformDataResponse(
              success=False,
              message="Missing 'columns' parameter for select operation.",
          )
        
        columns = request.parameters["columns"]
        df_transformed = df[columns]
      
      elif operation_lower == "sort":
        # Sort the data
        if "by" not in request.parameters:
          return TransformDataResponse(
              success=False,
              message="Missing 'by' parameter for sort operation.",
          )
        
        by = request.parameters["by"]
        ascending = request.parameters.get("ascending", True)
        df_transformed = df.sort_values(by=by, ascending=ascending)
      
      elif operation_lower == "group":
        # Group the data
        if "by" not in request.parameters:
          return TransformDataResponse(
              success=False,
              message="Missing 'by' parameter for group operation.",
          )
        
        by = request.parameters["by"]
        agg = request.parameters.get("agg", "mean")
        df_transformed = df.groupby(by).agg(agg).reset_index()
      
      else:
        return TransformDataResponse(
            success=False,
            message=f"Unsupported transformation operation '{request.operation}'.",
        )
      
      # Generate a unique ID for the transformed data
      data_id = f"data_{len(self._data_store) + 1}"
      self._data_store[data_id] = df_transformed
      
      return TransformDataResponse(
          success=True,
          message=f"Successfully transformed data using '{request.operation}'.",
          data_id=data_id,
      )
    except Exception as e:
      logger.exception("Error transforming data")
      return TransformDataResponse(
          success=False,
          message=f"Error transforming data: {str(e)}",
      )
  
  async def preprocess_data(
      self, request: DataPreprocessRequest
  ) -> DataPreprocessResponse:
    """Preprocess data.
    
    Args:
      request: The request containing the data ID and preprocessing operation.
      
    Returns:
      A response indicating whether the data was preprocessed successfully.
    """
    return await self._preprocessing_tools.preprocess_data(request)
  
  async def analyze_data(self, request: AnalyzeDataRequest) -> AnalyzeDataResponse:
    """Analyze data.
    
    Args:
      request: The request containing the data ID and analysis type.
      
    Returns:
      A response containing the analysis results.
    """
    try:
      # Check if the data ID exists
      if request.data_id not in self._data_store:
        return AnalyzeDataResponse(
            success=False,
            message=f"Data ID '{request.data_id}' not found.",
        )
      
      # If analysis_type is "auto" or "automatic", detect the best analysis type
      if request.analysis_type.lower() in ["auto", "automatic"]:
        # Detect analysis types
        detect_response = await self._analysis_detector.detect_analysis_type(
            DetectAnalysisTypeRequest(data_id=request.data_id)
        )
        
        if not detect_response.success:
          return AnalyzeDataResponse(
              success=False,
              message=f"Error detecting analysis types: {detect_response.message}",
          )
        
        # Return the detected analysis types and objectives
        return AnalyzeDataResponse(
            success=True,
            message="Successfully detected analysis types and objectives.",
            results={
                "dataset_profile": detect_response.dataset_profile.dict() if detect_response.dataset_profile else None,
                "detected_analysis_types": detect_response.detected_analysis_types,
                "analysis_objectives": [obj.dict() for obj in detect_response.analysis_objectives],
            },
        )
      
      # Check if the analysis type is in the list of allowed analysis types
      if self.analysis_types and request.analysis_type not in self.analysis_types:
        return AnalyzeDataResponse(
            success=False,
            message=f"Analysis type '{request.analysis_type}' is not in the list of allowed types.",
        )
      
      # Get the data
      df = self._data_store[request.data_id]
      
      # Perform the analysis
      analysis_type_lower = request.analysis_type.lower()
      results = {}
      
      if analysis_type_lower == "summary":
        # Generate a summary of the data
        results["shape"] = df.shape
        results["columns"] = df.columns.tolist()
        results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        results["missing_values"] = df.isnull().sum().to_dict()
        results["numeric_summary"] = df.describe().to_dict()
      
      elif analysis_type_lower == "correlation":
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
          return AnalyzeDataResponse(
              success=False,
              message="No numeric columns found for correlation analysis.",
          )
        
        results["correlation_matrix"] = numeric_df.corr().to_dict()
      
      elif analysis_type_lower == "distribution":
        # Analyze the distribution of a column
        if "column" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'column' parameter for distribution analysis.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        if pd.api.types.is_numeric_dtype(df[column]):
          results["min"] = float(df[column].min())
          results["max"] = float(df[column].max())
          results["mean"] = float(df[column].mean())
          results["median"] = float(df[column].median())
          results["std"] = float(df[column].std())
          results["quantiles"] = {
              "25%": float(df[column].quantile(0.25)),
              "50%": float(df[column].quantile(0.5)),
              "75%": float(df[column].quantile(0.75)),
          }
        else:
          results["value_counts"] = df[column].value_counts().to_dict()
          results["unique_values"] = df[column].nunique()
      
      elif analysis_type_lower == "outliers":
        # Detect outliers in a column
        if "column" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'column' parameter for outlier detection.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[column]):
          return AnalyzeDataResponse(
              success=False,
              message=f"Column '{column}' is not numeric.",
          )
        
        method = request.parameters.get("method", "zscore")
        threshold = request.parameters.get("threshold", 3)
        
        if method == "zscore":
          z_scores = abs((df[column] - df[column].mean()) / df[column].std())
          outliers = df[z_scores > threshold]
          results["outlier_count"] = len(outliers)
          results["outlier_percentage"] = len(outliers) / len(df) * 100
          results["outlier_indices"] = outliers.index.tolist()
          results["outlier_values"] = outliers[column].tolist()
        
        elif method == "iqr":
          q1 = df[column].quantile(0.25)
          q3 = df[column].quantile(0.75)
          iqr = q3 - q1
          lower_bound = q1 - 1.5 * iqr
          upper_bound = q3 + 1.5 * iqr
          outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
          results["outlier_count"] = len(outliers)
          results["outlier_percentage"] = len(outliers) / len(df) * 100
          results["outlier_indices"] = outliers.index.tolist()
          results["outlier_values"] = outliers[column].tolist()
          results["lower_bound"] = float(lower_bound)
          results["upper_bound"] = float(upper_bound)
        
        else:
          return AnalyzeDataResponse(
              success=False,
              message=f"Unsupported outlier detection method '{method}'.",
          )
      
      elif analysis_type_lower == "time_series":
        # Analyze time series data
        if "date_column" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'date_column' parameter for time series analysis.",
          )
        
        date_column = request.parameters["date_column"]
        if date_column not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Column '{date_column}' not found in the data.",
          )
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
          try:
            df[date_column] = pd.to_datetime(df[date_column])
          except:
            return AnalyzeDataResponse(
                success=False,
                message=f"Could not convert column '{date_column}' to datetime.",
            )
        
        # Sort by date
        df = df.sort_values(by=date_column)
        
        # Calculate time-based statistics
        results["start_date"] = df[date_column].min().isoformat()
        results["end_date"] = df[date_column].max().isoformat()
        results["duration"] = (df[date_column].max() - df[date_column].min()).days
        
        # Calculate frequency if value_column is provided
        if "value_column" in request.parameters:
          value_column = request.parameters["value_column"]
          if value_column not in df.columns:
            return AnalyzeDataResponse(
                success=False,
                message=f"Column '{value_column}' not found in the data.",
            )
          
          if not pd.api.types.is_numeric_dtype(df[value_column]):
            return AnalyzeDataResponse(
                success=False,
                message=f"Column '{value_column}' is not numeric.",
            )
          
          # Calculate trend
          df["index"] = range(len(df))
          import numpy as np
          from scipy import stats
          slope, intercept, r_value, p_value, std_err = stats.linregress(
              df["index"], df[value_column]
          )
          
          results["trend"] = {
              "slope": float(slope),
              "intercept": float(intercept),
              "r_squared": float(r_value ** 2),
              "p_value": float(p_value),
              "std_err": float(std_err),
          }
          
          # Calculate seasonality if enough data points
          if len(df) >= 12:
            # Simple seasonality detection using autocorrelation
            from pandas.plotting import autocorrelation_plot
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            autocorrelation_plot(df[value_column])
            
            # Save the plot
            seasonality_plot_path = os.path.join(self._temp_dir, "seasonality_plot.png")
            plt.savefig(seasonality_plot_path)
            plt.close()
            
            results["seasonality_plot_path"] = seasonality_plot_path
      
      elif analysis_type_lower == "clustering":
        # Perform clustering analysis
        if "columns" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'columns' parameter for clustering analysis.",
          )
        
        columns = request.parameters["columns"]
        for col in columns:
          if col not in df.columns:
            return AnalyzeDataResponse(
                success=False,
                message=f"Column '{col}' not found in the data.",
            )
          
          if not pd.api.types.is_numeric_dtype(df[col]):
            return AnalyzeDataResponse(
                success=False,
                message=f"Column '{col}' is not numeric.",
            )
        
        n_clusters = request.parameters.get("n_clusters", 3)
        
        # Perform K-means clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = df.groupby("cluster").agg({
            **{col: ["mean", "std", "min", "max"] for col in columns},
            "cluster": "count",
        })
        
        results["cluster_centers"] = {
            f"cluster_{i}": {
                col: float(val) for col, val in zip(columns, center)
            } for i, center in enumerate(kmeans.cluster_centers_)
        }
        
        results["cluster_counts"] = df["cluster"].value_counts().to_dict()
        results["cluster_stats"] = cluster_stats.to_dict()
        
        # Create a scatter plot of the first two dimensions
        if len(columns) >= 2:
          import matplotlib.pyplot as plt
          
          plt.figure(figsize=(10, 8))
          for i in range(n_clusters):
            plt.scatter(
                df[df["cluster"] == i][columns[0]],
                df[df["cluster"] == i][columns[1]],
                label=f"Cluster {i}",
            )
          
          plt.xlabel(columns[0])
          plt.ylabel(columns[1])
          plt.title("Cluster Analysis")
          plt.legend()
          
          # Save the plot
          cluster_plot_path = os.path.join(self._temp_dir, "cluster_plot.png")
          plt.savefig(cluster_plot_path)
          plt.close()
          
          results["cluster_plot_path"] = cluster_plot_path
      
      elif analysis_type_lower == "regression":
        # Perform regression analysis
        if "target" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'target' parameter for regression analysis.",
          )
        
        if "features" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'features' parameter for regression analysis.",
          )
        
        target = request.parameters["target"]
        features = request.parameters["features"]
        
        if target not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Target column '{target}' not found in the data.",
          )
        
        for feature in features:
          if feature not in df.columns:
            return AnalyzeDataResponse(
                success=False,
                message=f"Feature column '{feature}' not found in the data.",
            )
        
        # Perform linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        
        # Prepare the data
        X = df[features]
        y = df[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results["coefficients"] = {
            feature: float(coef) for feature, coef in zip(features, model.coef_)
        }
        results["intercept"] = float(model.intercept_)
        results["mse"] = float(mse)
        results["r2"] = float(r2)
        results["rmse"] = float(mse ** 0.5)
        
        # Create a scatter plot of actual vs predicted values
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Values")
        
        # Add a diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
        
        # Save the plot
        regression_plot_path = os.path.join(self._temp_dir, "regression_plot.png")
        plt.savefig(regression_plot_path)
        plt.close()
        
        results["regression_plot_path"] = regression_plot_path
      
      elif analysis_type_lower == "classification":
        # Perform classification analysis
        if "target" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'target' parameter for classification analysis.",
          )
        
        if "features" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'features' parameter for classification analysis.",
          )
        
        target = request.parameters["target"]
        features = request.parameters["features"]
        
        if target not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Target column '{target}' not found in the data.",
          )
        
        for feature in features:
          if feature not in df.columns:
            return AnalyzeDataResponse(
                success=False,
                message=f"Feature column '{feature}' not found in the data.",
            )
        
        # Perform logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        
        # Prepare the data
        X = df[features]
        y = df[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results["accuracy"] = float(accuracy)
        results["confusion_matrix"] = conf_matrix.tolist()
        results["classification_report"] = class_report
        
        # Create a confusion matrix plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        # Save the plot
        confusion_matrix_path = os.path.join(self._temp_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        results["confusion_matrix_path"] = confusion_matrix_path
      
      elif analysis_type_lower == "text_analysis":
        # Perform text analysis
        if "text_column" not in request.parameters:
          return AnalyzeDataResponse(
              success=False,
              message="Missing 'text_column' parameter for text analysis.",
          )
        
        text_column = request.parameters["text_column"]
        if text_column not in df.columns:
          return AnalyzeDataResponse(
              success=False,
              message=f"Column '{text_column}' not found in the data.",
          )
        
        # Calculate basic text statistics
        df["text_length"] = df[text_column].str.len()
        df["word_count"] = df[text_column].str.split().str.len()
        
        results["text_length"] = {
            "mean": float(df["text_length"].mean()),
            "median": float(df["text_length"].median()),
            "min": float(df["text_length"].min()),
            "max": float(df["text_length"].max()),
        }
        
        results["word_count"] = {
            "mean": float(df["word_count"].mean()),
            "median": float(df["word_count"].median()),
            "min": float(df["word_count"].min()),
            "max": float(df["word_count"].max()),
        }
        
        # Perform word frequency analysis
        from collections import Counter
        import re
        
        # Combine all text
        all_text = " ".join(df[text_column].fillna(""))
        
        # Tokenize and count words
        words = re.findall(r"\b\w+\b", all_text.lower())
        word_counts = Counter(words)
        
        # Get the top 20 words
        top_words = word_counts.most_common(20)
        
        results["top_words"] = {word: count for word, count in top_words}
        
        # Create a word cloud
        try:
          from wordcloud import WordCloud
          import matplotlib.pyplot as plt
          
          wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
          
          plt.figure(figsize=(10, 8))
          plt.imshow(wordcloud, interpolation="bilinear")
          plt.axis("off")
          
          # Save the plot
          wordcloud_path = os.path.join(self._temp_dir, "wordcloud.png")
          plt.savefig(wordcloud_path)
          plt.close()
          
          results["wordcloud_path"] = wordcloud_path
        except ImportError:
          results["wordcloud_error"] = "WordCloud package not installed."
      
      elif analysis_type_lower == "automatic":
        # Automatically determine the best analysis based on the data
        results["automatic_analysis"] = "Automatic analysis performed."
        
        # Perform basic summary analysis
        results["shape"] = df.shape
        results["columns"] = df.columns.tolist()
        results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        results["missing_values"] = df.isnull().sum().to_dict()
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
          results["numeric_summary"] = df[numeric_cols].describe().to_dict()
          
          # Calculate correlation matrix if multiple numeric columns
          if len(numeric_cols) > 1:
            results["correlation_matrix"] = df[numeric_cols].corr().to_dict()
        
        # Analyze categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
          results["categorical_summary"] = {
              col: {
                  "unique_values": df[col].nunique(),
                  "top_values": df[col].value_counts().head(5).to_dict(),
              }
              for col in cat_cols
          }
        
        # Check for datetime columns
        date_cols = []
        for col in df.columns:
          if pd.api.types.is_datetime64_dtype(df[col]):
            date_cols.append(col)
          else:
            try:
              pd.to_datetime(df[col])
              date_cols.append(col)
            except:
              pass
        
        if date_cols:
          results["date_columns"] = date_cols
          
          # Perform time series analysis on the first date column
          date_col = date_cols[0]
          
          # Convert to datetime if not already
          if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
          
          # Sort by date
          df_sorted = df.sort_values(by=date_col)
          
          results["time_series"] = {
              "start_date": df_sorted[date_col].min().isoformat(),
              "end_date": df_sorted[date_col].max().isoformat(),
              "duration": (df_sorted[date_col].max() - df_sorted[date_col].min()).days,
          }
      
      else:
        return AnalyzeDataResponse(
            success=False,
            message=f"Unsupported analysis type '{request.analysis_type}'.",
        )
      
      return AnalyzeDataResponse(
          success=True,
          message=f"Successfully performed {request.analysis_type} analysis.",
          results=results,
      )
    except Exception as e:
      logger.exception("Error analyzing data")
      return AnalyzeDataResponse(
          success=False,
          message=f"Error analyzing data: {str(e)}",
      )
  
  async def visualize_data(
      self, request: VisualizeDataRequest
  ) -> VisualizeDataResponse:
    """Visualize data.
    
    Args:
      request: The request containing the data ID and visualization type.
      
    Returns:
      A response containing the path to the visualization file.
    """
    try:
      # Check if the data ID exists
      if request.data_id not in self._data_store:
        return VisualizeDataResponse(
            success=False,
            message=f"Data ID '{request.data_id}' not found.",
        )
      
      # Check if the visualization type is in the list of allowed visualization types
      if self.visualization_types and request.visualization_type not in self.visualization_types:
        return VisualizeDataResponse(
            success=False,
            message=f"Visualization type '{request.visualization_type}' is not in the list of allowed types.",
        )
      
      # Get the data
      df = self._data_store[request.data_id]
      
      # Import matplotlib and seaborn
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      # Set the style
      sns.set(style="whitegrid")
      
      # Create the visualization
      visualization_type_lower = request.visualization_type.lower()
      
      if visualization_type_lower == "bar":
        # Create a bar chart
        if "x" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'x' parameter for bar chart.",
          )
        
        x = request.parameters["x"]
        if x not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{x}' not found in the data.",
          )
        
        y = request.parameters.get("y")
        if y and y not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{y}' not found in the data.",
          )
        
        plt.figure(figsize=(10, 6))
        
        if y:
          sns.barplot(x=x, y=y, data=df)
        else:
          sns.countplot(x=x, data=df)
        
        plt.title(request.parameters.get("title", f"Bar Chart of {x}"))
        plt.xlabel(request.parameters.get("xlabel", x))
        plt.ylabel(request.parameters.get("ylabel", y if y else "Count"))
        plt.xticks(rotation=request.parameters.get("rotation", 0))
        
        if "limit" in request.parameters:
          plt.xlim(0, request.parameters["limit"])
      
      elif visualization_type_lower == "line":
        # Create a line chart
        if "x" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'x' parameter for line chart.",
          )
        
        if "y" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'y' parameter for line chart.",
          )
        
        x = request.parameters["x"]
        y = request.parameters["y"]
        
        if x not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{x}' not found in the data.",
          )
        
        if y not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{y}' not found in the data.",
          )
        
        plt.figure(figsize=(10, 6))
        
        # Sort by x if it's a datetime column
        if pd.api.types.is_datetime64_dtype(df[x]) or pd.api.types.is_numeric_dtype(df[x]):
          df = df.sort_values(by=x)
        
        sns.lineplot(x=x, y=y, data=df)
        
        plt.title(request.parameters.get("title", f"Line Chart of {y} vs {x}"))
        plt.xlabel(request.parameters.get("xlabel", x))
        plt.ylabel(request.parameters.get("ylabel", y))
        plt.xticks(rotation=request.parameters.get("rotation", 0))
      
      elif visualization_type_lower == "scatter":
        # Create a scatter plot
        if "x" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'x' parameter for scatter plot.",
          )
        
        if "y" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'y' parameter for scatter plot.",
          )
        
        x = request.parameters["x"]
        y = request.parameters["y"]
        
        if x not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{x}' not found in the data.",
          )
        
        if y not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{y}' not found in the data.",
          )
        
        plt.figure(figsize=(10, 6))
        
        hue = request.parameters.get("hue")
        if hue and hue in df.columns:
          sns.scatterplot(x=x, y=y, hue=hue, data=df)
        else:
          sns.scatterplot(x=x, y=y, data=df)
        
        plt.title(request.parameters.get("title", f"Scatter Plot of {y} vs {x}"))
        plt.xlabel(request.parameters.get("xlabel", x))
        plt.ylabel(request.parameters.get("ylabel", y))
      
      elif visualization_type_lower == "histogram":
        # Create a histogram
        if "column" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'column' parameter for histogram.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[column]):
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' is not numeric.",
          )
        
        plt.figure(figsize=(10, 6))
        
        bins = request.parameters.get("bins", 10)
        kde = request.parameters.get("kde", True)
        
        sns.histplot(df[column], bins=bins, kde=kde)
        
        plt.title(request.parameters.get("title", f"Histogram of {column}"))
        plt.xlabel(request.parameters.get("xlabel", column))
        plt.ylabel(request.parameters.get("ylabel", "Frequency"))
      
      elif visualization_type_lower == "box":
        # Create a box plot
        if "column" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'column' parameter for box plot.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[column]):
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' is not numeric.",
          )
        
        plt.figure(figsize=(10, 6))
        
        by = request.parameters.get("by")
        if by and by in df.columns:
          sns.boxplot(x=by, y=column, data=df)
          plt.xticks(rotation=request.parameters.get("rotation", 0))
        else:
          sns.boxplot(y=column, data=df)
        
        plt.title(request.parameters.get("title", f"Box Plot of {column}"))
        if by:
          plt.xlabel(request.parameters.get("xlabel", by))
        plt.ylabel(request.parameters.get("ylabel", column))
      
      elif visualization_type_lower == "heatmap":
        # Create a heatmap
        if "columns" not in request.parameters:
          # Use all numeric columns
          columns = df.select_dtypes(include=["number"]).columns.tolist()
          if not columns:
            return VisualizeDataResponse(
                success=False,
                message="No numeric columns found for heatmap.",
            )
        else:
          columns = request.parameters["columns"]
          for col in columns:
            if col not in df.columns:
              return VisualizeDataResponse(
                  success=False,
                  message=f"Column '{col}' not found in the data.",
              )
            
            if not pd.api.types.is_numeric_dtype(df[col]):
              return VisualizeDataResponse(
                  success=False,
                  message=f"Column '{col}' is not numeric.",
              )
        
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr = df[columns].corr()
        
        # Create heatmap
        sns.heatmap(
            corr,
            annot=request.parameters.get("annot", True),
            cmap=request.parameters.get("cmap", "coolwarm"),
            fmt=request.parameters.get("fmt", ".2f"),
        )
        
        plt.title(request.parameters.get("title", "Correlation Heatmap"))
      
      elif visualization_type_lower == "pie":
        # Create a pie chart
        if "column" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'column' parameter for pie chart.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        plt.figure(figsize=(10, 8))
        
        # Get value counts
        value_counts = df[column].value_counts()
        
        # Limit the number of categories if specified
        max_categories = request.parameters.get("max_categories")
        if max_categories and len(value_counts) > max_categories:
          other_count = value_counts.iloc[max_categories:].sum()
          value_counts = value_counts.iloc[:max_categories]
          value_counts["Other"] = other_count
        
        # Create pie chart
        plt.pie(
            value_counts,
            labels=value_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            shadow=request.parameters.get("shadow", False),
        )
        
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(request.parameters.get("title", f"Pie Chart of {column}"))
      
      elif visualization_type_lower == "pair":
        # Create a pair plot
        if "columns" not in request.parameters:
          # Use all numeric columns
          columns = df.select_dtypes(include=["number"]).columns.tolist()
          if not columns:
            return VisualizeDataResponse(
                success=False,
                message="No numeric columns found for pair plot.",
            )
        else:
          columns = request.parameters["columns"]
          for col in columns:
            if col not in df.columns:
              return VisualizeDataResponse(
                  success=False,
                  message=f"Column '{col}' not found in the data.",
              )
            
            if not pd.api.types.is_numeric_dtype(df[col]):
              return VisualizeDataResponse(
                  success=False,
                  message=f"Column '{col}' is not numeric.",
              )
        
        # Limit the number of columns to avoid excessive computation
        max_columns = request.parameters.get("max_columns", 5)
        if len(columns) > max_columns:
          columns = columns[:max_columns]
        
        # Create pair plot
        hue = request.parameters.get("hue")
        if hue and hue in df.columns:
          g = sns.pairplot(df[columns + [hue]], hue=hue)
        else:
          g = sns.pairplot(df[columns])
        
        plt.suptitle(
            request.parameters.get("title", "Pair Plot"),
            y=1.02,
        )
      
      elif visualization_type_lower == "violin":
        # Create a violin plot
        if "column" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'column' parameter for violin plot.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[column]):
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' is not numeric.",
          )
        
        plt.figure(figsize=(10, 6))
        
        by = request.parameters.get("by")
        if by and by in df.columns:
          sns.violinplot(x=by, y=column, data=df)
          plt.xticks(rotation=request.parameters.get("rotation", 0))
        else:
          sns.violinplot(y=column, data=df)
        
        plt.title(request.parameters.get("title", f"Violin Plot of {column}"))
        if by:
          plt.xlabel(request.parameters.get("xlabel", by))
        plt.ylabel(request.parameters.get("ylabel", column))
      
      elif visualization_type_lower == "count":
        # Create a count plot
        if "column" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'column' parameter for count plot.",
          )
        
        column = request.parameters["column"]
        if column not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{column}' not found in the data.",
          )
        
        plt.figure(figsize=(10, 6))
        
        hue = request.parameters.get("hue")
        if hue and hue in df.columns:
          sns.countplot(x=column, hue=hue, data=df)
        else:
          sns.countplot(x=column, data=df)
        
        plt.title(request.parameters.get("title", f"Count Plot of {column}"))
        plt.xlabel(request.parameters.get("xlabel", column))
        plt.ylabel(request.parameters.get("ylabel", "Count"))
        plt.xticks(rotation=request.parameters.get("rotation", 0))
      
      elif visualization_type_lower == "joint":
        # Create a joint plot
        if "x" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'x' parameter for joint plot.",
          )
        
        if "y" not in request.parameters:
          return VisualizeDataResponse(
              success=False,
              message="Missing 'y' parameter for joint plot.",
          )
        
        x = request.parameters["x"]
        y = request.parameters["y"]
        
        if x not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{x}' not found in the data.",
          )
        
        if y not in df.columns:
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{y}' not found in the data.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[x]):
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{x}' is not numeric.",
          )
        
        if not pd.api.types.is_numeric_dtype(df[y]):
          return VisualizeDataResponse(
              success=False,
              message=f"Column '{y}' is not numeric.",
          )
        
        kind = request.parameters.get("kind", "scatter")
        g = sns.jointplot(
            x=x,
            y=y,
            data=df,
            kind=kind,
        )
        
        g.fig.suptitle(
            request.parameters.get("title", f"Joint Plot of {y} vs {x}"),
            y=1.02,
        )
        g.set_axis_labels(
            request.parameters.get("xlabel", x),
            request.parameters.get("ylabel", y),
        )
      
      else:
        return VisualizeDataResponse(
            success=False,
            message=f"Unsupported visualization type '{request.visualization_type}'.",
        )
      
      # Save the visualization
      visualization_path = os.path.join(
          self._temp_dir, f"{visualization_type_lower}_plot.png"
      )
      plt.savefig(visualization_path, bbox_inches="tight")
      plt.close()
      
      return VisualizeDataResponse(
          success=True,
          message=f"Successfully created {request.visualization_type} visualization.",
          visualization_path=visualization_path,
      )
    except Exception as e:
      logger.exception("Error visualizing data")
      return VisualizeDataResponse(
          success=False,
          message=f"Error visualizing data: {str(e)}",
      )

