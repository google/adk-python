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

"""Example usage of the Data Analysis Agent."""

import asyncio
import os
import pandas as pd
import tempfile

from google.adk.agents.templates.data_analysis import DataAnalysisAgent


async def main():
  """Run the example."""
  # Create a sample dataset
  df = pd.DataFrame({
      "numeric_col": [1, 2, None, 4, 5, 100],  # Contains missing value and outlier
      "categorical_col": ["A", "B", "A", None, "C", "B"],  # Contains missing value
      "date_col": pd.date_range(start="2023-01-01", periods=6, freq="D"),
  })
  
  # Save the dataset to a temporary file
  temp_dir = tempfile.mkdtemp()
  data_path = os.path.join(temp_dir, "sample_data.csv")
  df.to_csv(data_path, index=False)
  
  # Create a data analysis agent
  agent = DataAnalysisAgent(
      name="data_analyst",
      model="gemini-1.5-pro",
      data_sources=[data_path],
      analysis_types=["summary", "correlation", "distribution", "outliers", "time_series"],
      visualization_types=["line", "bar", "scatter", "histogram", "boxplot", "heatmap", "pie"],
      preprocessing_operations=[
          "clean_missing_values", "handle_outliers", "engineer_features", 
          "encode_categorical", "normalize_data", "remove_duplicates", "convert_types"
      ]
  )
  
  # Example 1: Load data and get a summary
  print("\n=== Example 1: Load data and get a summary ===")
  response = await agent.generate_content(
      f"Load data from {data_path} and provide a summary of the data."
  )
  print(response.text)
  
  # Example 2: Preprocess data - handle missing values
  print("\n=== Example 2: Preprocess data - handle missing values ===")
  response = await agent.generate_content(
      "The data has missing values. Please clean them by filling numeric columns with their mean values and categorical columns with their mode."
  )
  print(response.text)
  
  # Example 3: Preprocess data - handle outliers
  print("\n=== Example 3: Preprocess data - handle outliers ===")
  response = await agent.generate_content(
      "The numeric_col has an outlier (value 100). Please handle it using the z-score method with a threshold of 2 and cap the outliers."
  )
  print(response.text)
  
  # Example 4: Preprocess data - engineer features
  print("\n=== Example 4: Preprocess data - engineer features ===")
  response = await agent.generate_content(
      "Extract date features from the date_col column, including year, month, day, and day of week."
  )
  print(response.text)
  
  # Example 5: Preprocess data - encode categorical
  print("\n=== Example 5: Preprocess data - encode categorical ===")
  response = await agent.generate_content(
      "Encode the categorical_col using one-hot encoding."
  )
  print(response.text)
  
  # Example 6: Preprocess data - normalize data
  print("\n=== Example 6: Preprocess data - normalize data ===")
  response = await agent.generate_content(
      "Normalize the numeric_col using min-max scaling to a range of [0, 1]."
  )
  print(response.text)
  
  # Example 7: Analyze data
  print("\n=== Example 7: Analyze data ===")
  response = await agent.generate_content(
      "Analyze the distribution of the numeric_col and categorical_col."
  )
  print(response.text)
  
  # Example 8: Visualize data
  print("\n=== Example 8: Visualize data ===")
  response = await agent.generate_content(
      "Create a histogram of the numeric_col and a pie chart of the categorical_col."
  )
  print(response.text)
  
  # Example 9: Complete data analysis workflow
  print("\n=== Example 9: Complete data analysis workflow ===")
  response = await agent.generate_content(
      f"""
      I need a complete data analysis workflow for the dataset at {data_path}:
      1. Load the data
      2. Clean missing values
      3. Handle outliers
      4. Engineer features from the date column
      5. Encode categorical variables
      6. Normalize numeric data
      7. Analyze the data
      8. Create visualizations
      9. Provide insights and recommendations
      """
  )
  print(response.text)


if __name__ == "__main__":
  asyncio.run(main())

