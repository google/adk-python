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

"""Analysis detector for the Data Analysis Agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalysisObjective(BaseModel):
    """Analysis objective detected from the data."""
    
    objective_type: str = Field(..., description="The type of analysis objective.")
    description: str = Field(..., description="Description of the analysis objective.")
    confidence: float = Field(..., description="Confidence score for the objective (0-1).")
    suggested_analyses: List[str] = Field(
        default_factory=list, description="Suggested analysis types for this objective."
    )
    suggested_visualizations: List[str] = Field(
        default_factory=list, description="Suggested visualization types for this objective."
    )
    relevant_columns: List[str] = Field(
        default_factory=list, description="Columns relevant to this objective."
    )


class DatasetProfile(BaseModel):
    """Profile of a dataset."""
    
    shape: Tuple[int, int] = Field(..., description="Shape of the dataset (rows, columns).")
    column_types: Dict[str, str] = Field(
        default_factory=dict, description="Types of columns in the dataset."
    )
    missing_values: Dict[str, int] = Field(
        default_factory=dict, description="Missing values in each column."
    )
    numeric_columns: List[str] = Field(
        default_factory=list, description="Numeric columns in the dataset."
    )
    categorical_columns: List[str] = Field(
        default_factory=list, description="Categorical columns in the dataset."
    )
    datetime_columns: List[str] = Field(
        default_factory=list, description="Datetime columns in the dataset."
    )
    text_columns: List[str] = Field(
        default_factory=list, description="Text columns in the dataset."
    )
    unique_values: Dict[str, int] = Field(
        default_factory=dict, description="Number of unique values in each column."
    )
    column_correlations: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Correlations between numeric columns."
    )


class DetectAnalysisTypeRequest(BaseModel):
    """Request to detect analysis types for a dataset."""
    
    data_id: str = Field(..., description="The identifier of the data to analyze.")


class DetectAnalysisTypeResponse(BaseModel):
    """Response from detecting analysis types."""
    
    success: bool = Field(..., description="Whether the detection was successful.")
    message: str = Field(..., description="A message describing the result.")
    dataset_profile: Optional[DatasetProfile] = Field(
        None, description="Profile of the dataset."
    )
    detected_analysis_types: List[str] = Field(
        default_factory=list, description="Detected analysis types for the dataset."
    )
    analysis_objectives: List[AnalysisObjective] = Field(
        default_factory=list, description="Detected analysis objectives for the dataset."
    )


class AnalysisDetector:
    """Detector for analysis types and objectives."""
    
    def __init__(self, data_store: Dict[str, pd.DataFrame]):
        """Initialize the analysis detector.
        
        Args:
            data_store: A dictionary mapping data IDs to DataFrames.
        """
        self._data_store = data_store
    
    async def detect_analysis_type(
        self, request: DetectAnalysisTypeRequest
    ) -> DetectAnalysisTypeResponse:
        """Detect analysis types for a dataset.
        
        Args:
            request: The request containing the data ID.
            
        Returns:
            A response containing the detected analysis types and objectives.
        """
        try:
            # Check if the data ID exists
            if request.data_id not in self._data_store:
                return DetectAnalysisTypeResponse(
                    success=False,
                    message=f"Data ID '{request.data_id}' not found.",
                )
            
            # Get the data
            df = self._data_store[request.data_id]
            
            # Create a profile of the dataset
            dataset_profile = self._create_dataset_profile(df)
            
            # Detect analysis types
            detected_analysis_types = self._detect_analysis_types(df, dataset_profile)
            
            # Detect analysis objectives
            analysis_objectives = self._detect_analysis_objectives(df, dataset_profile)
            
            return DetectAnalysisTypeResponse(
                success=True,
                message="Successfully detected analysis types and objectives.",
                dataset_profile=dataset_profile,
                detected_analysis_types=detected_analysis_types,
                analysis_objectives=analysis_objectives,
            )
        except Exception as e:
            logger.exception("Error detecting analysis types")
            return DetectAnalysisTypeResponse(
                success=False,
                message=f"Error detecting analysis types: {str(e)}",
            )
    
    def _create_dataset_profile(self, df: pd.DataFrame) -> DatasetProfile:
        """Create a profile of the dataset.
        
        Args:
            df: The DataFrame to profile.
            
        Returns:
            A profile of the dataset.
        """
        # Get the shape of the dataset
        shape = df.shape
        
        # Get the types of columns
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Get missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Identify column types
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
        
        # Identify potential datetime columns that are not already detected
        for col in df.columns:
            if col not in datetime_columns:
                try:
                    pd.to_datetime(df[col])
                    datetime_columns.append(col)
                    if col in categorical_columns:
                        categorical_columns.remove(col)
                except:
                    pass
        
        # Identify text columns (categorical columns with long text)
        text_columns = []
        for col in categorical_columns:
            # Check if the column contains long text (more than 100 characters on average)
            if df[col].astype(str).str.len().mean() > 100:
                text_columns.append(col)
        
        # Remove text columns from categorical columns
        categorical_columns = [col for col in categorical_columns if col not in text_columns]
        
        # Get unique values
        unique_values = {col: df[col].nunique() for col in df.columns}
        
        # Calculate correlations between numeric columns
        column_correlations = {}
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr().to_dict()
            column_correlations = corr_matrix
        
        return DatasetProfile(
            shape=shape,
            column_types=column_types,
            missing_values=missing_values,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            text_columns=text_columns,
            unique_values=unique_values,
            column_correlations=column_correlations,
        )
    
    def _detect_analysis_types(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> List[str]:
        """Detect analysis types for a dataset.
        
        Args:
            df: The DataFrame to analyze.
            profile: The profile of the dataset.
            
        Returns:
            A list of detected analysis types.
        """
        analysis_types = ["summary"]  # Always include summary analysis
        
        # Check for correlation analysis
        if len(profile.numeric_columns) > 1:
            analysis_types.append("correlation")
        
        # Check for distribution analysis
        if profile.numeric_columns:
            analysis_types.append("distribution")
        
        # Check for outlier analysis
        if profile.numeric_columns:
            analysis_types.append("outliers")
        
        # Check for time series analysis
        if profile.datetime_columns:
            analysis_types.append("time_series")
        
        # Check for clustering analysis
        if len(profile.numeric_columns) >= 2:
            analysis_types.append("clustering")
        
        # Check for regression analysis
        if len(profile.numeric_columns) >= 2:
            analysis_types.append("regression")
        
        # Check for classification analysis
        if profile.categorical_columns and profile.numeric_columns:
            # Check if any categorical column has a small number of unique values
            for col in profile.categorical_columns:
                if profile.unique_values.get(col, 0) <= 10:
                    analysis_types.append("classification")
                    break
        
        # Check for text analysis
        if profile.text_columns:
            analysis_types.append("text_analysis")
        
        return analysis_types
    
    def _detect_analysis_objectives(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> List[AnalysisObjective]:
        """Detect analysis objectives for a dataset.
        
        Args:
            df: The DataFrame to analyze.
            profile: The profile of the dataset.
            
        Returns:
            A list of detected analysis objectives.
        """
        objectives = []
        
        # Check for data quality objective
        if any(count > 0 for count in profile.missing_values.values()):
            missing_columns = [col for col, count in profile.missing_values.items() if count > 0]
            missing_percentage = sum(profile.missing_values.values()) / (profile.shape[0] * profile.shape[1]) * 100
            
            objectives.append(
                AnalysisObjective(
                    objective_type="data_quality",
                    description=f"Improve data quality by addressing missing values in {len(missing_columns)} columns ({missing_percentage:.2f}% of all values).",
                    confidence=min(1.0, missing_percentage / 10),
                    suggested_analyses=["summary"],
                    suggested_visualizations=["bar", "heatmap"],
                    relevant_columns=missing_columns,
                )
            )
        
        # Check for correlation analysis objective
        if len(profile.numeric_columns) > 1 and profile.column_correlations:
            # Find highly correlated pairs
            high_corr_pairs = []
            for col1, corrs in profile.column_correlations.items():
                for col2, corr in corrs.items():
                    if col1 != col2 and abs(corr) > 0.7:
                        high_corr_pairs.append((col1, col2, abs(corr)))
            
            if high_corr_pairs:
                high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = high_corr_pairs[:3]
                
                pair_descriptions = [f"{col1} and {col2} (correlation: {corr:.2f})" for col1, col2, corr in top_pairs]
                relevant_columns = list(set([col for pair in top_pairs for col in pair[:2]]))
                
                objectives.append(
                    AnalysisObjective(
                        objective_type="correlation_analysis",
                        description=f"Analyze relationships between variables, particularly {', '.join(pair_descriptions)}.",
                        confidence=min(1.0, max(pair[2] for pair in top_pairs)),
                        suggested_analyses=["correlation", "regression"],
                        suggested_visualizations=["scatter", "heatmap", "pair"],
                        relevant_columns=relevant_columns,
                    )
                )
        
        # Check for time series analysis objective
        if profile.datetime_columns:
            datetime_col = profile.datetime_columns[0]
            
            # Check if there are numeric columns that could be analyzed over time
            if profile.numeric_columns:
                objectives.append(
                    AnalysisObjective(
                        objective_type="time_series_analysis",
                        description=f"Analyze trends and patterns over time using the datetime column '{datetime_col}'.",
                        confidence=0.8,
                        suggested_analyses=["time_series"],
                        suggested_visualizations=["line", "bar"],
                        relevant_columns=[datetime_col] + profile.numeric_columns[:3],
                    )
                )
        
        # Check for clustering objective
        if len(profile.numeric_columns) >= 3:
            objectives.append(
                AnalysisObjective(
                    objective_type="clustering",
                    description=f"Identify natural groupings or segments in the data based on {len(profile.numeric_columns)} numeric features.",
                    confidence=min(1.0, len(profile.numeric_columns) / 10),
                    suggested_analyses=["clustering"],
                    suggested_visualizations=["scatter", "pair"],
                    relevant_columns=profile.numeric_columns[:5],  # Limit to top 5 columns
                )
            )
        
        # Check for predictive modeling objective
        if len(profile.numeric_columns) >= 2 and profile.categorical_columns:
            # Look for potential target variables
            potential_targets = []
            
            # Check categorical columns with few unique values
            for col in profile.categorical_columns:
                if 2 <= profile.unique_values.get(col, 0) <= 10:
                    potential_targets.append((col, "classification"))
            
            # Check numeric columns that might be targets
            for col in profile.numeric_columns:
                # Check if column name suggests it's a target (e.g., contains 'price', 'sales', 'revenue', etc.)
                target_keywords = ["price", "sales", "revenue", "income", "cost", "profit", "target", "value"]
                if any(keyword in col.lower() for keyword in target_keywords):
                    potential_targets.append((col, "regression"))
            
            if potential_targets:
                target, model_type = potential_targets[0]  # Take the first potential target
                
                objectives.append(
                    AnalysisObjective(
                        objective_type="predictive_modeling",
                        description=f"Build a {model_type} model to predict '{target}' based on other features.",
                        confidence=0.7,
                        suggested_analyses=[model_type],
                        suggested_visualizations=["scatter", "box", "bar"] if model_type == "classification" else ["scatter", "line", "histogram"],
                        relevant_columns=[target] + [col for col in profile.numeric_columns + profile.categorical_columns if col != target][:5],
                    )
                )
        
        # Check for outlier detection objective
        if profile.numeric_columns:
            objectives.append(
                AnalysisObjective(
                    objective_type="outlier_detection",
                    description=f"Identify and analyze outliers in numeric columns that may affect analysis results.",
                    confidence=0.6,
                    suggested_analyses=["outliers", "distribution"],
                    suggested_visualizations=["box", "histogram", "scatter"],
                    relevant_columns=profile.numeric_columns[:5],  # Limit to top 5 columns
                )
            )
        
        # Check for text analysis objective
        if profile.text_columns:
            objectives.append(
                AnalysisObjective(
                    objective_type="text_analysis",
                    description=f"Analyze text content in {len(profile.text_columns)} columns to extract insights and patterns.",
                    confidence=0.7,
                    suggested_analyses=["text_analysis"],
                    suggested_visualizations=["bar", "count"],
                    relevant_columns=profile.text_columns,
                )
            )
        
        return objectives

