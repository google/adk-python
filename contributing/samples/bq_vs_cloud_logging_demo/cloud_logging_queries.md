# Cloud Logging Queries for ADK Agents

This document shows how to query ADK agent telemetry data using Cloud Logging.

## Prerequisites

- Cloud Logging API enabled
- Agent running with `--otel_to_cloud` flag
- Log name: `adk-otel` (default)

## Basic Log Queries

### View All ADK Logs

```
logName="projects/PROJECT_ID/logs/adk-otel"
```

### Filter by Severity

```
logName="projects/PROJECT_ID/logs/adk-otel"
severity>=WARNING
```

### Filter by Time Range

```
logName="projects/PROJECT_ID/logs/adk-otel"
timestamp>="2025-01-01T00:00:00Z"
timestamp<="2025-01-02T00:00:00Z"
```

## Trace-Based Queries

### Find Logs for a Specific Trace

```
logName="projects/PROJECT_ID/logs/adk-otel"
trace="projects/PROJECT_ID/traces/TRACE_ID"
```

### View Logs with Span Context

```
logName="projects/PROJECT_ID/logs/adk-otel"
spanId="SPAN_ID"
```

## Limitations for Analytics

Cloud Logging is designed for operational monitoring, not deep analytics. Here's
what's **challenging** to achieve compared to BigQuery:

### 1. Token Usage Analysis

**Not directly available in Cloud Logging.**

Token usage data isn't captured in standard OTel spans. You would need to:
- Parse log messages for token counts (if your agent logs them)
- Use Cloud Monitoring custom metrics instead
- Export to BigQuery for analysis

```
# Best effort: Search for any logs mentioning tokens
logName="projects/PROJECT_ID/logs/adk-otel"
textPayload:"token"
```

### 2. Tool Failure Rate

**Requires manual log parsing and aggregation.**

```
# Find tool errors
logName="projects/PROJECT_ID/logs/adk-otel"
severity=ERROR
textPayload:"tool"

# Find tool completions
logName="projects/PROJECT_ID/logs/adk-otel"
textPayload:"tool" AND textPayload:"completed"
```

To calculate failure rates, you must:
1. Export logs to BigQuery using a Log Sink
2. Run SQL queries on the exported data
3. Or use Log-based Metrics (count-based only, limited dimensions)

### 3. Multi-Modal Content Analysis

**Very limited support.**

Cloud Logging has size limits (~256KB per entry) and doesn't natively handle
binary content like images. Multi-modal analysis requires:
- Content stored elsewhere (GCS)
- Only metadata logged
- External joins for full analysis

## Cloud Logging to BigQuery Export (Workaround)

If you need analytics capabilities, create a Log Sink to export to BigQuery:

```bash
# Create a BigQuery dataset for logs
bq mk --dataset PROJECT_ID:adk_logs

# Create a Log Sink
gcloud logging sinks create adk-to-bigquery \
  bigquery.googleapis.com/projects/PROJECT_ID/datasets/adk_logs \
  --log-filter='logName="projects/PROJECT_ID/logs/adk-otel"'
```

Then query the exported data:

```sql
SELECT
  timestamp,
  severity,
  textPayload,
  jsonPayload
FROM `PROJECT_ID.adk_logs.adk_otel_*`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 100;
```

**Note:** Exported logs still lack the structured schema that the BigQuery
Agent Analytics plugin provides (token usage, tool metadata, multimodal parts).

## Log-Based Metrics (Alternative)

Create counter metrics for basic monitoring:

```bash
# Create a metric for tool errors
gcloud logging metrics create tool_errors \
  --description="Count of tool errors in ADK agents" \
  --log-filter='logName="projects/PROJECT_ID/logs/adk-otel" AND severity=ERROR AND textPayload:"tool"'
```

These metrics appear in Cloud Monitoring but lack the granularity of SQL analytics.

## Cloud Trace Integration

The primary value of Cloud Logging with OTel is **distributed tracing**:

1. View traces in Cloud Console:
   ```
   https://console.cloud.google.com/traces/list?project=PROJECT_ID
   ```

2. Correlate logs with traces for debugging
3. Visualize request flow across services

This is where Cloud Logging excels over the BigQuery plugin alone.

## Summary

| Analysis Type | Cloud Logging Capability |
|--------------|-------------------------|
| Token Usage | Not available |
| Tool Failure Rates | Manual parsing required |
| Multi-Modal Content | Very limited |
| Session Analytics | Basic (requires export) |
| Distributed Tracing | Excellent |
| Real-time Debugging | Excellent |
| Alerting | Good (via Log-based alerts) |

For comprehensive analytics, use the **BigQuery Agent Analytics Plugin**.
For operational monitoring and tracing, use **Cloud Logging via OTel**.
