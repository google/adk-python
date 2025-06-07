# Data Analysis Agent Template

This template provides a specialized agent for data analysis tasks, including data loading, preprocessing, transformation, analysis, and visualization.

## Features

- **Multiple Data Sources**: Load data from various sources including file uploads, Google Sheets, databases, and APIs.
- **Comprehensive Preprocessing**: Clean and prepare data with operations for handling missing values, outliers, categorical encoding, and more.
- **Automatic Analysis Detection**: Automatically detect appropriate analysis types and objectives based on dataset characteristics.
- **Rich Analysis Capabilities**: Perform various types of analysis including summary statistics, correlation analysis, time series analysis, clustering, regression, and more.
- **Visualization Generation**: Create visualizations to illustrate insights, with support for various chart types.

## Usage

```python
from google.adk.agents.templates.data_analysis.data_analysis_agent import DataAnalysisAgent

# Create a Data Analysis Agent
agent = DataAnalysisAgent(
    name="data_analysis_agent",
    model="gemini-1.5-pro",
    description="A specialized agent for data analysis tasks.",
)

# Use the agent to analyze data
response = await agent.generate_content("Analyze the sales data in sales.csv and identify trends.")
```

## Data Sources

The agent supports loading data from various sources:

- **File Upload**: Load data from local files (CSV, Excel, JSON, etc.)
- **Google Sheets**: Connect to and load data from Google Sheets
- **Databases**: Connect to databases (MySQL, PostgreSQL, SQLite, etc.)
- **APIs**: Fetch data from APIs

## Preprocessing Operations

The agent supports various preprocessing operations:

- **clean_missing_values**: Handle missing values in the data (drop, fill with mean/median/mode/constant)
- **handle_outliers**: Detect and handle outliers (z-score, IQR, percentile methods)
- **engineer_features**: Create new features (polynomial, interaction, binning, date features)
- **encode_categorical**: Encode categorical variables (one-hot, label, ordinal, target encoding)
- **normalize_data**: Normalize or standardize numeric data (min-max, z-score, robust scaling)
- **remove_duplicates**: Remove duplicate rows from the data
- **convert_types**: Convert column data types (numeric, datetime, category)

## Analysis Types

The agent supports various types of analysis:

- **summary**: Generate basic statistics and summary of the data
- **correlation**: Calculate correlation matrix between numeric columns
- **distribution**: Analyze the distribution of values in a column
- **outliers**: Detect outliers in numeric columns
- **time_series**: Analyze time-based patterns and trends
- **clustering**: Identify natural groupings in the data
- **regression**: Build predictive models for numeric targets
- **classification**: Build predictive models for categorical targets
- **text_analysis**: Analyze text content and extract insights
- **automatic**: Automatically detect the best analysis types for the data

## Visualization Types

The agent supports various types of visualizations:

- **bar**: Create bar charts for categorical data
- **line**: Create line charts for time series or trends
- **scatter**: Create scatter plots to show relationships between variables
- **histogram**: Create histograms to show distributions
- **box**: Create box plots to show distributions and outliers
- **heatmap**: Create heatmaps to show correlations or patterns
- **pie**: Create pie charts to show proportions
- **pair**: Create pair plots to show relationships between multiple variables
- **violin**: Create violin plots to show distributions
- **count**: Create count plots for categorical data
- **joint**: Create joint plots to show distributions and relationships

## Automatic Analysis Detection

The agent can automatically detect the best analysis types and objectives for a dataset. This feature:

1. Creates a profile of the dataset (column types, missing values, etc.)
2. Detects suitable analysis types based on the data characteristics
3. Identifies potential analysis objectives (e.g., correlation analysis, time series analysis, clustering)
4. Suggests relevant columns for each analysis objective
5. Recommends appropriate visualizations for each objective

## Example

See [data_analysis_example.py](../../examples/data_analysis_example.py) for a complete example of using the Data Analysis Agent.

## Customization

You can customize the agent by specifying:

- **data_sources**: List of data sources to analyze
- **analysis_types**: List of analysis types to perform
- **visualization_types**: List of visualization types to generate
- **preprocessing_operations**: List of preprocessing operations to perform
- **memory_service**: Optional memory service for persisting analysis results

```python
from google.adk.agents.templates.data_analysis.data_analysis_agent import DataAnalysisAgent

# Create a customized Data Analysis Agent
agent = DataAnalysisAgent(
    name="custom_data_analysis_agent",
    model="gemini-1.5-pro",
    description="A customized agent for financial data analysis.",
    data_sources=["file_upload", "database"],
    analysis_types=["summary", "correlation", "time_series"],
    visualization_types=["line", "bar", "scatter"],
    preprocessing_operations=["clean_missing_values", "handle_outliers"],
)
```

