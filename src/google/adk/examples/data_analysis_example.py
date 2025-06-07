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

"""Example of using the Data Analysis Agent."""

import asyncio
import logging
import os
import pandas as pd
import tempfile

from google.adk.agents.templates.data_analysis.data_analysis_agent import DataAnalysisAgent
from google.adk.agents.templates.data_analysis.data_analysis_tools import (
    FileUploadRequest, FileUploadResponse,
    GoogleSheetRequest, GoogleSheetResponse,
    DatabaseConnectionRequest, DatabaseConnectionResponse,
    APIDataRequest, APIDataResponse,
    DetectAnalysisTypeRequest, DetectAnalysisTypeResponse,
    DataPreprocessRequest, DataPreprocessResponse,
    AnalyzeDataRequest, AnalyzeDataResponse,
    VisualizeDataRequest, VisualizeDataResponse,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run the example."""
    # Create a temporary directory for data files
    temp_dir = tempfile.mkdtemp()
    
    # Create a sample dataset
    sample_data_path = os.path.join(temp_dir, "sample_data.csv")
    create_sample_dataset(sample_data_path)
    
    # Create the Data Analysis Agent
    agent = DataAnalysisAgent(
        name="data_analysis_agent",
        model="gemini-1.5-pro",
        description="A specialized agent for data analysis tasks.",
    )
    
    # Get the data analysis toolset
    toolset = agent.tools[0]
    
    # Example 1: Upload a file
    logger.info("Example 1: Uploading a file")
    upload_response = await toolset.upload_file(
        FileUploadRequest(
            file_path=sample_data_path,
            file_type="csv",
        )
    )
    
    if upload_response.success:
        logger.info(f"File uploaded successfully. Data ID: {upload_response.data_id}")
        data_id = upload_response.data_id
    else:
        logger.error(f"File upload failed: {upload_response.message}")
        return
    
    # Example 2: Detect analysis types
    logger.info("\nExample 2: Detecting analysis types")
    detect_response = await toolset.detect_analysis_type(
        DetectAnalysisTypeRequest(
            data_id=data_id,
        )
    )
    
    if detect_response.success:
        logger.info("Analysis types detected successfully.")
        logger.info(f"Dataset profile: {detect_response.dataset_profile}")
        logger.info(f"Detected analysis types: {detect_response.detected_analysis_types}")
        logger.info(f"Analysis objectives: {detect_response.analysis_objectives}")
    else:
        logger.error(f"Analysis type detection failed: {detect_response.message}")
    
    # Example 3: Preprocess data
    logger.info("\nExample 3: Preprocessing data")
    preprocess_response = await toolset.preprocess_data(
        DataPreprocessRequest(
            data_id=data_id,
            operations=[
                {
                    "operation": "clean_missing_values",
                    "columns": ["age", "income"],
                    "method": "mean",
                },
                {
                    "operation": "handle_outliers",
                    "columns": ["age", "income"],
                    "method": "zscore",
                    "threshold": 3,
                },
                {
                    "operation": "encode_categorical",
                    "columns": ["gender", "education"],
                    "method": "one_hot",
                },
            ],
        )
    )
    
    if preprocess_response.success:
        logger.info(f"Data preprocessed successfully. Data ID: {preprocess_response.data_id}")
        preprocessed_data_id = preprocess_response.data_id
    else:
        logger.error(f"Data preprocessing failed: {preprocess_response.message}")
        preprocessed_data_id = data_id
    
    # Example 4: Analyze data
    logger.info("\nExample 4: Analyzing data")
    analyze_response = await toolset.analyze_data(
        AnalyzeDataRequest(
            data_id=preprocessed_data_id,
            analysis_type="correlation",
            parameters={},
        )
    )
    
    if analyze_response.success:
        logger.info("Data analyzed successfully.")
        logger.info(f"Analysis results: {analyze_response.results}")
    else:
        logger.error(f"Data analysis failed: {analyze_response.message}")
    
    # Example 5: Visualize data
    logger.info("\nExample 5: Visualizing data")
    visualize_response = await toolset.visualize_data(
        VisualizeDataRequest(
            data_id=preprocessed_data_id,
            visualization_type="scatter",
            parameters={
                "x": "age",
                "y": "income",
                "title": "Age vs Income",
            },
        )
    )
    
    if visualize_response.success:
        logger.info("Data visualized successfully.")
        logger.info(f"Visualization path: {visualize_response.visualization_path}")
    else:
        logger.error(f"Data visualization failed: {visualize_response.message}")
    
    # Example 6: Automatic analysis
    logger.info("\nExample 6: Automatic analysis")
    auto_analyze_response = await toolset.analyze_data(
        AnalyzeDataRequest(
            data_id=preprocessed_data_id,
            analysis_type="automatic",
            parameters={},
        )
    )
    
    if auto_analyze_response.success:
        logger.info("Automatic analysis completed successfully.")
        logger.info(f"Dataset profile: {auto_analyze_response.results.get('dataset_profile')}")
        logger.info(f"Detected analysis types: {auto_analyze_response.results.get('detected_analysis_types')}")
        logger.info(f"Analysis objectives: {auto_analyze_response.results.get('analysis_objectives')}")
    else:
        logger.error(f"Automatic analysis failed: {auto_analyze_response.message}")
    
    logger.info("\nExample completed successfully.")


def create_sample_dataset(file_path):
    """Create a sample dataset for the example.
    
    Args:
        file_path: Path to save the sample dataset.
    """
    # Create a sample dataset
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a dataframe with 100 rows
    n = 100
    
    # Create demographic data
    age = np.random.normal(35, 10, n).astype(int)
    gender = np.random.choice(["Male", "Female"], n)
    education = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n)
    
    # Create financial data with some correlation to age
    base_income = 30000 + age * 1000
    income = base_income + np.random.normal(0, 10000, n)
    
    # Create purchase data
    purchase_frequency = np.random.poisson(5, n)
    avg_purchase_value = 50 + np.random.exponential(50, n)
    
    # Create customer satisfaction data
    satisfaction = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.4, 0.25])
    
    # Create date data
    start_date = pd.Timestamp("2023-01-01")
    dates = [start_date + pd.Timedelta(days=int(d)) for d in np.random.randint(0, 365, n)]
    
    # Create a dataframe
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": age,
        "gender": gender,
        "education": education,
        "income": income,
        "purchase_frequency": purchase_frequency,
        "avg_purchase_value": avg_purchase_value,
        "satisfaction": satisfaction,
        "last_purchase_date": dates,
    })
    
    # Add some missing values
    for col in ["age", "income", "purchase_frequency", "avg_purchase_value"]:
        mask = np.random.choice([True, False], n, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(range(n), 3, replace=False)
    df.loc[outlier_indices[0], "age"] = 90
    df.loc[outlier_indices[1], "income"] = 500000
    df.loc[outlier_indices[2], "avg_purchase_value"] = 1000
    
    # Save the dataframe to a CSV file
    df.to_csv(file_path, index=False)
    logger.info(f"Sample dataset created at {file_path}")


if __name__ == "__main__":
    asyncio.run(main())

