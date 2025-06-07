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

"""Data Analysis Agent implementation for ADK."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import Field
from typing_extensions import override

from google.adk.agents.llm_agent import LlmAgent, ToolUnion
from google.adk.memory.base_memory_service import BaseMemoryService

from .data_analysis_tools import DataAnalysisToolset


class DataAnalysisAgent(LlmAgent):
  """Specialized agent for data analysis tasks.
  
  This agent extends the base LlmAgent with specialized capabilities for data
  analysis, including tools for data loading, preprocessing, transformation, 
  analysis, and visualization.
  
  Attributes:
    data_sources: List of data sources to analyze.
    analysis_types: List of analysis types to perform.
    visualization_types: List of visualization types to generate.
    preprocessing_operations: List of preprocessing operations to perform.
    memory_service: Optional memory service for persisting analysis results.
  """
  
  data_sources: List[str] = Field(default_factory=list)
  """List of data sources to analyze."""
  
  analysis_types: List[str] = Field(default_factory=list)
  """List of analysis types to perform."""
  
  visualization_types: List[str] = Field(default_factory=list)
  """List of visualization types to generate."""
  
  preprocessing_operations: List[str] = Field(default_factory=list)
  """List of preprocessing operations to perform."""
  
  memory_service: Optional[BaseMemoryService] = None
  """Optional memory service for persisting analysis results."""
  
  def __init__(
      self,
      *,
      name: str,
      model: str,
      description: str = "A specialized agent for data analysis tasks.",
      data_sources: Optional[List[str]] = None,
      analysis_types: Optional[List[str]] = None,
      visualization_types: Optional[List[str]] = None,
      preprocessing_operations: Optional[List[str]] = None,
      memory_service: Optional[BaseMemoryService] = None,
      tools: Optional[List[ToolUnion]] = None,
      **kwargs: Any,
  ) -> None:
    """Initialize a DataAnalysisAgent.
    
    Args:
      name: The name of the agent.
      model: The model to use for the agent.
      description: Description about the agent's capability.
      data_sources: List of data sources to analyze.
      analysis_types: List of analysis types to perform.
      visualization_types: List of visualization types to generate.
      preprocessing_operations: List of preprocessing operations to perform.
      memory_service: Optional memory service for persisting analysis results.
      tools: Additional tools to provide to the agent.
      **kwargs: Additional arguments to pass to the parent class.
    """
    # Create default instruction if not provided
    if "instruction" not in kwargs:
      kwargs["instruction"] = self._create_default_instruction(
          data_sources, analysis_types, visualization_types, preprocessing_operations
      )
    
    # Initialize data sources, analysis types, and visualization types
    self.data_sources = data_sources or []
    self.analysis_types = analysis_types or []
    self.visualization_types = visualization_types or []
    self.preprocessing_operations = preprocessing_operations or [
        "clean_missing_values", "handle_outliers", "engineer_features", 
        "encode_categorical", "normalize_data", "remove_duplicates", "convert_types"
    ]
    self.memory_service = memory_service
    
    # Create data analysis toolset
    data_analysis_toolset = DataAnalysisToolset(
        data_sources=self.data_sources,
        analysis_types=self.analysis_types,
        visualization_types=self.visualization_types,
        memory_service=self.memory_service,
    )
    
    # Combine with additional tools if provided
    all_tools = [data_analysis_toolset]
    if tools:
      all_tools.extend(tools)
    
    # Initialize parent class
    super().__init__(
        name=name,
        model=model,
        description=description,
        tools=all_tools,
        **kwargs,
    )
  
  def _create_default_instruction(
      self,
      data_sources: Optional[List[str]],
      analysis_types: Optional[List[str]],
      visualization_types: Optional[List[str]],
      preprocessing_operations: Optional[List[str]],
  ) -> str:
    """Create a default instruction for the agent based on the provided parameters.
    
    Args:
      data_sources: List of data sources to analyze.
      analysis_types: List of analysis types to perform.
      visualization_types: List of visualization types to generate.
      preprocessing_operations: List of preprocessing operations to perform.
      
    Returns:
      A default instruction string.
    """
    instruction = (
        "You are a specialized data analysis assistant. "
        "Your goal is to help users analyze data, extract insights, "
        "and create visualizations. "
        "Follow these guidelines:\n\n"
        
        "1. Understand the user's data analysis needs clearly before proceeding.\n"
        "2. Use the provided tools to load, preprocess, transform, analyze, and visualize data.\n"
        "3. Always consider data quality issues and apply appropriate preprocessing steps.\n"
        "4. Explain your analysis approach and findings in clear, concise language.\n"
        "5. When presenting results, include both the raw data and your interpretation.\n"
        "6. For visualizations, explain what the visualization shows and why it's useful.\n"
        "7. If you encounter limitations or need more information, ask the user.\n\n"
    )
    
    # Add data sources information
    instruction += "Available data sources:\n"
    default_sources = [
        "file_upload: Upload local files (CSV, Excel, JSON, etc.)",
        "google_sheet: Connect to Google Sheets",
        "database: Connect to databases (MySQL, PostgreSQL, SQLite, etc.)",
        "api: Fetch data from APIs"
    ]
    
    if data_sources:
        for source in data_sources:
            instruction += f"- {source}\n"
    else:
        for source in default_sources:
            instruction += f"- {source}\n"
    instruction += "\n"
    
    # Add preprocessing operations information
    preprocessing_ops = preprocessing_operations or [
        "clean_missing_values", "handle_outliers", "engineer_features", 
        "encode_categorical", "normalize_data", "remove_duplicates", "convert_types"
    ]
    
    preprocessing_descriptions = {
        "clean_missing_values": "Handle missing values in the data (drop, fill with mean/median/mode/constant)",
        "handle_outliers": "Detect and handle outliers (z-score, IQR, percentile methods)",
        "engineer_features": "Create new features (polynomial, interaction, binning, date features)",
        "encode_categorical": "Encode categorical variables (one-hot, label, ordinal, target encoding)",
        "normalize_data": "Normalize or standardize numeric data (min-max, z-score, robust scaling)",
        "remove_duplicates": "Remove duplicate rows from the data",
        "convert_types": "Convert column data types (numeric, datetime, category)"
    }
    
    instruction += "Available preprocessing operations:\n"
    for op in preprocessing_ops:
        description = preprocessing_descriptions.get(op, op)
        instruction += f"- {op}: {description}\n"
    instruction += "\n"
    
    # Add analysis types information
    default_analysis_types = [
        "summary: Generate basic statistics and summary of the data",
        "correlation: Calculate correlation matrix between numeric columns",
        "distribution: Analyze the distribution of values in a column",
        "outliers: Detect outliers in numeric columns",
        "time_series: Analyze time-based patterns and trends",
        "clustering: Identify natural groupings in the data",
        "regression: Build predictive models for numeric targets",
        "classification: Build predictive models for categorical targets",
        "text_analysis: Analyze text content and extract insights",
        "automatic: Automatically detect the best analysis types for the data"
    ]
    
    instruction += "Available analysis types:\n"
    if analysis_types:
        for analysis_type in analysis_types:
            instruction += f"- {analysis_type}\n"
    else:
        for analysis_type in default_analysis_types:
            instruction += f"- {analysis_type}\n"
    instruction += "\n"
    
    # Add visualization types information
    default_visualization_types = [
        "bar: Create bar charts for categorical data",
        "line: Create line charts for time series or trends",
        "scatter: Create scatter plots to show relationships between variables",
        "histogram: Create histograms to show distributions",
        "box: Create box plots to show distributions and outliers",
        "heatmap: Create heatmaps to show correlations or patterns",
        "pie: Create pie charts to show proportions",
        "pair: Create pair plots to show relationships between multiple variables",
        "violin: Create violin plots to show distributions",
        "count: Create count plots for categorical data",
        "joint: Create joint plots to show distributions and relationships"
    ]
    
    instruction += "Available visualization types:\n"
    if visualization_types:
        for viz_type in visualization_types:
            instruction += f"- {viz_type}\n"
    else:
        for viz_type in default_visualization_types:
            instruction += f"- {viz_type}\n"
    instruction += "\n"
    
    # Add automatic analysis detection information
    instruction += (
        "Automatic Analysis Detection:\n"
        "You can automatically detect the best analysis types and objectives for a dataset using the "
        "`detect_analysis_type` method or by specifying `analysis_type=\"automatic\"` in the `analyze_data` method. "
        "This will:\n"
        "1. Create a profile of the dataset (column types, missing values, etc.)\n"
        "2. Detect suitable analysis types based on the data characteristics\n"
        "3. Identify potential analysis objectives (e.g., correlation analysis, time series analysis, clustering)\n"
        "4. Suggest relevant columns for each analysis objective\n"
        "5. Recommend appropriate visualizations for each objective\n\n"
    )
    
    # Add data analysis workflow guidance
    instruction += (
        "Recommended data analysis workflow:\n"
        "1. Load data from the appropriate source (file upload, Google Sheet, database, API)\n"
        "2. Explore the data to understand its structure and quality\n"
        "3. Use automatic analysis detection to identify potential analysis objectives\n"
        "4. Preprocess the data to handle missing values, outliers, and other quality issues\n"
        "5. Transform the data as needed for analysis (filtering, selecting, grouping)\n"
        "6. Perform analysis to extract insights\n"
        "7. Create visualizations to illustrate key findings\n"
        "8. Summarize results and provide recommendations\n\n"
    )
    
    return instruction

