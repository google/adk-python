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

"""Data Analysis Agent using CodingAgent.

This sample demonstrates a CodingAgent configured as a data analyst that can:
- Fetch datasets from URLs (CSV, JSON, text)
- Analyze data using pandas
- Create visualizations using matplotlib
- Generate statistical summaries and insights

Prerequisites:
- Docker must be installed and running
- Set GOOGLE_API_KEY or configure Vertex AI credentials

Usage:
    adk run contributing/samples/coding_agent
    adk web contributing/samples

Example queries:
- "What is the survival rate on the Titanic?"
- "Create a bar chart showing survival rate by passenger class"
- "Analyze the iris dataset and create a scatter plot"
"""

import base64
import binascii
import os
import urllib.error
import urllib.request
from datetime import datetime

from google.adk.agents import CodingAgent
from google.adk.code_executors import ContainerCodeExecutor
from google.adk.code_executors.allowlist_validator import DEFAULT_SAFE_IMPORTS


# Sample dataset URLs
SAMPLE_DATASETS = {
    "titanic": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "description": "Titanic passenger data with survival information. 891 passengers with features like age, sex, class, fare, and survival status.",
        "columns": "PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked",
    },
    "iris": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "description": "Iris flower dataset. 150 samples of 3 species with sepal and petal measurements.",
        "columns": "sepal_length, sepal_width, petal_length, petal_width, species",
    },
    "tips": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "description": "Restaurant tips dataset. 244 records with bill amount, tip, and customer info.",
        "columns": "total_bill, tip, sex, smoker, day, time, size",
    },
}


def fetch_url(url: str) -> dict:
    """Fetch content from a URL.

    Fetches data from the specified URL and returns the content along with
    metadata. Supports CSV, JSON, and plain text content.

    Args:
        url: The URL to fetch content from.

    Returns:
        Dictionary containing:
        - content: The fetched content as a string
        - content_type: The MIME type of the content
        - size: Size of the content in bytes
        - url: The original URL
        - success: Whether the fetch was successful
        - error: Error message if fetch failed (only present on failure)
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ADK-DataAnalyst/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "text/plain")
            return {
                "content": content,
                "content_type": content_type,
                "size": len(content),
                "url": url,
                "success": True,
            }
    except urllib.error.URLError as e:
        return {
            "content": "",
            "url": url,
            "success": False,
            "error": f"Failed to fetch URL: {str(e)}",
        }
    except Exception as e:
        return {
            "content": "",
            "url": url,
            "success": False,
            "error": f"Unexpected error: {str(e)}",
        }


def get_sample_datasets() -> dict:
    """Get available sample datasets with their URLs and descriptions.

    Returns a dictionary of sample datasets that can be used for analysis.
    Each dataset includes a URL, description, and column information.

    Returns:
        Dictionary with dataset names as keys, each containing:
        - url: Direct URL to download the CSV file
        - description: Brief description of the dataset
        - columns: Comma-separated list of column names
    """
    return SAMPLE_DATASETS


def get_current_time() -> dict:
    """Get the current date and time.

    Returns:
        Dictionary containing:
        - timestamp: ISO format timestamp
        - year, month, day: Date components
        - hour, minute, second: Time components
        - weekday: Name of the day of the week
    """
    now = datetime.now()
    return {
        "timestamp": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "weekday": now.strftime("%A"),
    }


# Directory on host system to save charts
HOST_CHARTS_DIR = "/tmp/adk_charts"


def save_chart(image_data: str, filename: str) -> dict:
    """Save a chart image to the host system.

    This tool saves base64-encoded image data to the host machine's filesystem,
    making charts accessible outside the Docker container.

    To use this tool, first save your matplotlib figure to a bytes buffer,
    then encode it as base64:

    Example:
        import base64
        import io
        import matplotlib.pyplot as plt

        # Create your plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        # Save to buffer and encode
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Save to host system
        result = save_chart(image_data=image_data, filename="my_chart.png")

    Args:
        image_data: Base64-encoded image data (PNG format recommended).
        filename: Name for the saved file (e.g., "chart.png").

    Returns:
        Dictionary containing:
        - success: Whether the save was successful
        - filepath: Full path where the file was saved on the host
        - size: Size of the saved file in bytes
        - error: Error message if save failed (only present on failure)
    """
    try:
        # Ensure the output directory exists
        os.makedirs(HOST_CHARTS_DIR, exist_ok=True)

        # Sanitize filename
        safe_filename = os.path.basename(filename)
        if not safe_filename:
            safe_filename = "chart.png"

        filepath = os.path.join(HOST_CHARTS_DIR, safe_filename)

        # Decode and save
        image_bytes = base64.b64decode(image_data)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return {
            "success": True,
            "filepath": filepath,
            "size": len(image_bytes),
            "message": f"Chart saved to {filepath}",
        }
    except binascii.Error as e:
        return {
            "success": False,
            "error": f"Invalid base64 data: {str(e)}",
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"Failed to save file: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
        }


def list_saved_charts() -> dict:
    """List all charts saved on the host system.

    Returns:
        Dictionary containing:
        - success: Whether the operation was successful
        - charts: List of saved chart filenames
        - directory: The directory where charts are saved
        - count: Number of charts found
    """
    try:
        if not os.path.exists(HOST_CHARTS_DIR):
            return {
                "success": True,
                "charts": [],
                "directory": HOST_CHARTS_DIR,
                "count": 0,
            }

        charts = [
            f
            for f in os.listdir(HOST_CHARTS_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf"))
        ]
        return {
            "success": True,
            "charts": charts,
            "directory": HOST_CHARTS_DIR,
            "count": len(charts),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list charts: {str(e)}",
        }


# Additional imports allowed for data analysis
DATA_ANALYSIS_IMPORTS = frozenset(
    {
        # Data analysis
        "pandas",
        "pandas.*",
        "numpy",
        "numpy.*",
        # Visualization
        "matplotlib",
        "matplotlib.*",
        "seaborn",
        "seaborn.*",
        # Data I/O
        "csv",
        "io",
        "io.*",
        # Encoding for chart saving
        "base64",
        # Subprocess for pip installs
        "subprocess",
    }
)


# Create the Data Analysis Agent
root_agent = CodingAgent(
    name="data_analyst",
    description=(
        "An AI data analyst that analyzes datasets, creates visualizations, "
        "and generates insights using Python code execution."
    ),
    model="gemini-2.5-flash",
    instruction="""You are a data analyst. Analyze data, create visualizations, and provide insights.

IMPORTANT: First install required packages before using them:
```tool_code
import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "matplotlib", "seaborn", "numpy"], check=True)
print("Packages installed successfully")
```

Then use the available tools to fetch datasets. Write Python code to analyze data using pandas and create charts with matplotlib.

CRITICAL: You MUST use the save_chart() tool to save charts - do NOT use plt.savefig() to a file path directly. The save_chart() tool transfers the chart to the host system. Here is the REQUIRED pattern:
```tool_code
import base64
import io
import matplotlib.pyplot as plt

# Create your plot
plt.figure(figsize=(10, 6))
# ... your plotting code ...

# Save to buffer and encode as base64
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Use the save_chart tool to save to host system
result = save_chart(image_data=image_data, filename="my_chart.png")
print(f"Chart saved: {result}")
```

The chart will be saved to the host system at /tmp/adk_charts/. Always report this filepath in your final answer.

Call final_answer() with your findings when done.
""",
    tools=[
        fetch_url,
        get_sample_datasets,
        get_current_time,
        save_chart,
        list_saved_charts,
    ],
    code_executor=ContainerCodeExecutor(
        image="python:3.11-slim",
    ),
    authorized_imports=DEFAULT_SAFE_IMPORTS | DATA_ANALYSIS_IMPORTS,
    max_iterations=10,
    error_retry_attempts=2,
    stateful=False,
)
