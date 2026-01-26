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

"""Conversational Data Analysis Agent using CodingAgent.

This sample demonstrates a CodingAgent configured as a conversational data
analyst that can:
- Have multi-turn conversations about data analysis
- Fetch datasets from URLs (CSV, JSON, text)
- Analyze data using pandas
- Create visualizations using matplotlib
- Generate statistical summaries and insights
- Remember context from previous messages

Prerequisites:
- Docker must be installed and running
- Set GOOGLE_API_KEY or configure Vertex AI credentials

Usage:
    adk run contributing/samples/coding_agent
    adk web contributing/samples

Example conversation:
    User: "What datasets do you have available?"
    Agent: "I have access to three sample datasets: titanic, iris, and tips..."

    User: "Let's look at the titanic data. What's the survival rate?"
    Agent: "I'll analyze the Titanic dataset for you... The overall survival
            rate was 38.4%. Would you like me to break this down by gender
            or passenger class?"

    User: "Yes, show me by passenger class with a chart"
    Agent: "Here's a bar chart showing survival rates by class... First class
            passengers had the highest survival rate at 63%..."
"""

import base64
import binascii
from datetime import datetime
import os
import socket
import urllib.error
import urllib.request

from google.adk.agents import CodingAgent
from google.adk.code_executors import ContainerCodeExecutor
from google.adk.code_executors.allowlist_validator import DEFAULT_SAFE_IMPORTS

# Sample dataset URLs
SAMPLE_DATASETS = {
    "titanic": {
        "url": (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        ),
        "description": (
            "Titanic passenger data with survival information. 891 passengers"
            " with features like age, sex, class, fare, and survival status."
        ),
        "columns": (
            "PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch,"
            " Ticket, Fare, Cabin, Embarked"
        ),
    },
    "iris": {
        "url": (
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        ),
        "description": (
            "Iris flower dataset. 150 samples of 3 species with sepal and petal"
            " measurements."
        ),
        "columns": ("sepal_length, sepal_width, petal_length, petal_width, species"),
    },
    "tips": {
        "url": (
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        ),
        "description": (
            "Restaurant tips dataset. 244 records with bill amount, tip, and"
            " customer info."
        ),
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
    except (urllib.error.URLError, socket.timeout) as e:
        return {
            "content": "",
            "url": url,
            "success": False,
            "error": f"Failed to fetch URL: {str(e)}",
        }
    except OSError as e:
        return {
            "content": "",
            "url": url,
            "success": False,
            "error": f"Failed to decode response: {str(e)}",
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
    except OSError as e:
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
    instruction="""You are a friendly, conversational data analyst assistant. You help users analyze datasets, create visualizations, and generate insights using Python code execution.

## Your Personality
- Be conversational and engaging - ask clarifying questions when needed
- Explain your analysis in plain language that anyone can understand
- Offer suggestions for follow-up analyses or visualizations
- Remember context from previous messages in our conversation

## When the user asks for analysis or visualization:

1. First, install required packages (only needed once per session):
```tool_code
import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "matplotlib", "seaborn", "numpy"], check=True)
print("Packages installed successfully")
```

2. Use the available tools:
   - `get_sample_datasets()` - See available sample datasets (titanic, iris, tips)
   - `fetch_url(url)` - Fetch data from any URL
   - `save_chart(image_data, filename)` - Save visualizations
   - `list_saved_charts()` - See previously saved charts

3. When creating charts, ALWAYS use this pattern to save them:
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

# Save to host system
result = save_chart(image_data=image_data, filename="descriptive_name.png")
print(f"Chart saved: {result}")
```

## Response Style
- After completing an analysis, summarize your findings in a conversational way
- Mention the chart location if you created one (charts are saved to /tmp/adk_charts/)
- Ask if the user would like to explore the data further or see different visualizations
- If you don't have enough information, ask questions before diving into code

## Examples of good responses:
- "I found some interesting patterns! The survival rate was 38.4%. Would you like me to break this down by gender or passenger class?"
- "Here's a bar chart showing the distribution. I saved it to /tmp/adk_charts/survival_by_class.png. What aspect would you like to explore next?"
- "Before I create that visualization, could you tell me which columns you're most interested in comparing?"

Remember: You're having a conversation, not just executing tasks. Engage with the user!
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
    stateful=True,
)
