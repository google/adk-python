# Agent Analytics Dashboard

A real-time monitoring dashboard for ADK agent behavior analytics stored in BigQuery. This dashboard provides comprehensive insights into agent performance, tool usage, error tracking, and session analysis.

## Features

- **Overview Dashboard**: Key metrics including sessions, events, error rates, latency, and token usage
- **Session Explorer**: Browse and inspect individual conversation sessions
- **Tool Performance**: Analyze tool invocation rates, success/failure metrics, and latency
- **LLM Metrics**: Track LLM requests, tokens, time-to-first-token, and latency trends
- **Error Tracking**: Aggregate error summaries with affected session counts
- **Real-time Updates**: Auto-refresh every 30 seconds

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ADK Agent     │────▶│   BigQuery      │◀────│   Dashboard     │
│   + Plugin      │     │   Analytics     │     │   (FastAPI)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
   Agent Events          Time-Partitioned        Interactive UI
   (LLM, Tools)          Clustered Table         with Charts
```

## Prerequisites

1. **Google Cloud Project** with BigQuery API enabled
2. **BigQuery Dataset** (the plugin auto-creates tables)
3. **Authentication**: Application Default Credentials (ADC) configured
4. **Python 3.11+**

## Quick Start

### 1. Set Environment Variables

```bash
export BQ_AGENT_ANALYTICS_PROJECT="your-gcp-project-id"
export BQ_AGENT_ANALYTICS_DATASET="your-bigquery-dataset"
export BQ_AGENT_ANALYTICS_TABLE="agent_events_v2"  # optional, default

# For Vertex AI (if using Gemini models)
export VERTEXAI_PROJECT="your-gcp-project-id"
export VERTEXAI_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### 2. Install Dependencies

```bash
cd contributing/samples/bigquery_agent_analytics_demo/dashboard
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
# Option 1: Using uvicorn directly
uvicorn app:app --reload --port 8080

# Option 2: Using Python
python app.py
```

### 4. Open in Browser

Navigate to: http://localhost:8080

## Generating Sample Data

Use the simulation script to populate your BigQuery table with realistic agent data:

```bash
# Simulate 10 conversation sessions
python simulate_agent_data.py --num-sessions 10

# Simulate with more prompts per session
python simulate_agent_data.py --num-sessions 20 --min-prompts 3 --max-prompts 8

# Generate historical data (multiple days worth)
python simulate_agent_data.py --historical --days 7 --sessions-per-day 50

# Disable error simulation
python simulate_agent_data.py --num-sessions 10 --no-errors
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML UI |
| `GET /api/health` | Health check |
| `GET /api/config` | Current configuration |
| `GET /api/overview?hours=24` | Overview metrics |
| `GET /api/events/types?hours=24` | Event type distribution |
| `GET /api/events/timeline?hours=24` | Events over time |
| `GET /api/sessions?hours=24&limit=50` | Session list |
| `GET /api/sessions/{session_id}/events` | Session event details |
| `GET /api/tools/metrics?hours=24` | Tool performance |
| `GET /api/llm/metrics?hours=24` | LLM performance |
| `GET /api/llm/latency-timeline?hours=24` | LLM latency over time |
| `GET /api/errors?hours=24` | Error summary |
| `GET /api/agents?hours=24` | Per-agent metrics |
| `GET /api/trace/{trace_id}` | Trace event details |

## BigQuery Table Schema

The dashboard reads from a table with this schema (auto-created by the plugin):

```sql
CREATE TABLE `project.dataset.agent_events_v2` (
  timestamp TIMESTAMP NOT NULL,
  event_type STRING,
  agent STRING,
  session_id STRING,
  invocation_id STRING,
  user_id STRING,
  trace_id STRING,
  span_id STRING,
  parent_span_id STRING,
  content JSON,
  content_parts ARRAY<STRUCT<
    mime_type STRING,
    uri STRING,
    object_ref STRUCT<uri STRING, version STRING, authorizer STRING, details JSON>,
    text STRING,
    part_index INT64,
    part_attributes STRING,
    storage_mode STRING
  >>,
  attributes JSON,
  latency_ms JSON,
  status STRING,
  error_message STRING,
  is_truncated BOOL
)
PARTITION BY DATE(timestamp)
CLUSTER BY event_type, agent, user_id;
```

## Deployment Options

### Local Development

```bash
uvicorn app:app --reload --port 8080
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t agent-analytics-dashboard .
docker run -p 8080:8080 \
  -e BQ_AGENT_ANALYTICS_PROJECT=your-project \
  -e BQ_AGENT_ANALYTICS_DATASET=your-dataset \
  -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json \
  -v /path/to/credentials.json:/path/to/credentials.json \
  agent-analytics-dashboard
```

### Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT/agent-analytics-dashboard

gcloud run deploy agent-analytics-dashboard \
  --image gcr.io/YOUR_PROJECT/agent-analytics-dashboard \
  --platform managed \
  --region us-central1 \
  --set-env-vars BQ_AGENT_ANALYTICS_PROJECT=your-project,BQ_AGENT_ANALYTICS_DATASET=your-dataset \
  --allow-unauthenticated
```

## Customization

### Adding Custom Metrics

Edit `app.py` to add new endpoints:

```python
@app.get("/api/custom/metric")
async def get_custom_metric(hours: int = 24):
    client = get_bq_client()
    query = f"""
    SELECT
      -- Your custom aggregation
    FROM {get_full_table_id()}
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    """
    result = client.query(query).result()
    return [dict(row) for row in result]
```

### Modifying the UI

The dashboard UI is embedded in `get_dashboard_html()`. Modify the HTML/JavaScript to:
- Add new charts (Chart.js)
- Change layouts (Tailwind CSS)
- Add new tabs or panels

## Troubleshooting

### "BigQuery configuration not set"
Ensure `BQ_AGENT_ANALYTICS_PROJECT` and `BQ_AGENT_ANALYTICS_DATASET` environment variables are set.

### "Permission denied"
Ensure your credentials have `roles/bigquery.dataViewer` and `roles/bigquery.jobUser` on the project/dataset.

### No data showing
1. Check that the table exists in BigQuery
2. Verify events are being logged (run the simulation script)
3. Adjust the time range filter in the UI

### Slow queries
- Ensure the table is partitioned by timestamp
- Add clustering on frequently filtered columns
- Consider materializing views for complex aggregations

## License

Apache 2.0 - See LICENSE file
