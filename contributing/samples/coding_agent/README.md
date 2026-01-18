# Data Analysis Agent

A CodingAgent sample that demonstrates AI-powered data analysis with Python code execution.

## Overview

This sample showcases the CodingAgent's ability to:

- Fetch datasets from URLs (CSV, JSON, text)
- Analyze data using pandas
- Create visualizations with matplotlib
- Generate statistical summaries and insights
- Execute multi-step reasoning through code

The agent writes and executes Python code in a sandboxed Docker container, calling tools via HTTP IPC when needed.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────>│   CodingAgent    │────>│ Docker Container│
│                 │     │  (Gemini 2.5)    │     │ (Python 3.11)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               │                         │ Executes
                               v                         │ pandas/matplotlib
                        ┌──────────────┐                 │ code
                        │ Tool Server  │<────────────────┘
                        │ (HTTP IPC)   │  Tool calls
                        └──────────────┘  (fetch_url, etc.)
```

### How It Works

1. User sends a natural language query
2. CodingAgent (powered by Gemini 2.5) generates Python code
3. Code is executed in a sandboxed Docker container
4. Tools (like `fetch_url`) are called via HTTP to the Tool Server on the host
5. Results are returned to the container, LLM iterates if needed
6. Final answer is provided via `final_answer()` function

## Prerequisites

- **Docker**: Must be installed and running
- **API Key**: Set `GOOGLE_API_KEY` in `.env` file or environment
- **Python**: 3.10+ (for running ADK CLI)

## Quick Start

### 1. Set up your API key

Create a `.env` file in this directory:

```bash
echo "GOOGLE_API_KEY=your_api_key_here" > contributing/samples/coding_agent/.env
```

### 2. Run the agent

**Using CLI (interactive):**

```bash
adk run contributing/samples/coding_agent
```

**Using Web UI:**

```bash
adk web contributing/samples
```

Then navigate to `http://localhost:8000` and select `coding_agent`.

## Available Tools

### `fetch_url(url: str) -> dict`

Fetches content from a URL. Supports CSV, JSON, and plain text.

**Returns:**
- `content`: The fetched content as a string
- `content_type`: MIME type of the content
- `size`: Size in bytes
- `success`: Boolean indicating success
- `error`: Error message (only on failure)

**Example:**
```python
result = fetch_url("https://example.com/data.csv")
if result["success"]:
    csv_content = result["content"]
```

### `get_sample_datasets() -> dict`

Returns available sample datasets with URLs and descriptions.

**Available datasets:**
- `titanic`: Titanic passenger survival data (891 rows)
- `iris`: Iris flower classification data (150 rows)
- `tips`: Restaurant tipping data (244 rows)

### `get_current_time() -> dict`

Returns current date and time information.

### `save_chart(image_data: str, filename: str) -> dict`

Saves a chart image to the **host system** (not the Docker container). This is essential for making visualizations accessible outside of Docker.

**Parameters:**
- `image_data`: Base64-encoded image data (PNG recommended)
- `filename`: Name for the saved file (e.g., "chart.png")

**Returns:**
- `success`: Boolean indicating success
- `filepath`: Full path where the file was saved (e.g., `/tmp/adk_charts/chart.png`)
- `size`: Size of the saved file in bytes
- `error`: Error message (only on failure)

**Example usage in generated code:**
```python
import base64
import io
import matplotlib.pyplot as plt

# Create your plot
plt.figure(figsize=(10, 6))
plt.bar(['A', 'B', 'C'], [1, 2, 3])

# Save to buffer and encode
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Save to host system
result = save_chart(image_data=image_data, filename="my_chart.png")
print(f"Chart saved to: {result['filepath']}")
```

### `list_saved_charts() -> dict`

Lists all charts saved to the host system.

**Returns:**
- `charts`: List of saved chart filenames
- `directory`: Directory path (`/tmp/adk_charts`)
- `count`: Number of charts found

## Sample Datasets

| Dataset | Description | Columns |
|---------|-------------|---------|
| **Titanic** | Passenger survival data from the Titanic disaster | PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked |
| **Iris** | Classic flower classification dataset | sepal_length, sepal_width, petal_length, petal_width, species |
| **Tips** | Restaurant tipping behavior data | total_bill, tip, sex, smoker, day, time, size |

## Example Interactions

### Basic Analysis

```
User: What is the survival rate on the Titanic?

Agent: [Fetches Titanic CSV, calculates survival statistics]
Response: The overall survival rate on the Titanic was 38.4% (342 out of 891 passengers survived).
```

### Visualization

```
User: Create a bar chart showing survival rate by passenger class on the Titanic

Agent: [Fetches data, creates matplotlib chart, uses save_chart tool]
Response: I've created a bar chart showing survival rates by class:
- 1st Class: 63% survival rate
- 2nd Class: 47% survival rate  
- 3rd Class: 24% survival rate
Chart saved to: /tmp/adk_charts/survival_by_class.png
```

Charts are saved to `/tmp/adk_charts/` on your host system, making them accessible outside Docker.

### Exploratory Analysis

```
User: Analyze the iris dataset and give me key insights

Agent: [Fetches iris data, runs statistical analysis, creates visualizations]
Response: Key insights from the Iris dataset:
1. The dataset contains 150 samples across 3 species (50 each)
2. Setosa is clearly separable from the other two species
3. Petal length is the best single predictor for species classification
...
```

### Custom URL

```
User: Fetch the tips dataset and tell me which day has the highest average tip

Agent: [Fetches tips CSV, analyzes by day]
Response: Sunday has the highest average tip at $3.26, followed by Saturday at $2.99.
```

## Troubleshooting

### Docker not running

```
Error: Cannot connect to Docker daemon
```

**Solution:** Start Docker Desktop or the Docker service.

### API key issues

```
Error: Missing key inputs argument!
```

**Solution:** Ensure `GOOGLE_API_KEY` is set in `.env` file or environment.

### Container networking issues

```
Error: Connection refused to tool server
```

**Solution:** On macOS/Windows, Docker Desktop should handle `host.docker.internal` automatically. On Linux, you may need to configure the Docker bridge network.

### Package installation in container

The agent automatically installs pandas and matplotlib at runtime. If you see import errors, ensure the container has internet access.

## Extending the Agent

### Adding Custom Tools

Add new tool functions in `agent.py`:

```python
def my_custom_tool(param: str) -> dict:
    """Description of what this tool does.
    
    Args:
        param: Description of the parameter.
        
    Returns:
        Dictionary with results.
    """
    # Implementation
    return {"result": "..."}

# Add to the tools list
root_agent = CodingAgent(
    ...
    tools=[fetch_url, get_sample_datasets, get_current_time, my_custom_tool],
    ...
)
```

### Using a Custom Container Image

For faster execution with pre-installed packages:

```python
code_executor=ContainerCodeExecutor(
    image="my-custom-image:latest",  # Image with pandas/matplotlib pre-installed
)
```

### Adjusting Iteration Limits

```python
root_agent = CodingAgent(
    ...
    max_iterations=15,        # More iterations for complex tasks
    error_retry_attempts=3,   # More retries on errors
    ...
)
```

## Related Documentation

- [CodingAgent Documentation](https://google.github.io/adk-docs/agents/coding-agent)
- [ContainerCodeExecutor](https://google.github.io/adk-docs/code-executors/container)
- [ADK Tools](https://google.github.io/adk-docs/tools)
