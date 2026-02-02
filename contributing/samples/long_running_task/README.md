# Durable Session Demo

This demo showcases the durable session persistence feature in ADK, which
enables checkpoint-based durability for long-running agent invocations.

## Overview

Durable sessions provide:
- **Checkpoint persistence**: Agent state is saved to BigQuery + GCS
- **Failure recovery**: Resume from the last checkpoint after crashes
- **Host migration**: Move sessions between hosts seamlessly
- **Lease management**: Prevent concurrent modifications

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **APIs enabled**:
   - BigQuery API
   - Cloud Storage API
   - Vertex AI API (for Gemini models)
3. **IAM permissions**:
   - `roles/bigquery.dataEditor`
   - `roles/storage.objectAdmin`
   - `roles/aiplatform.user`

## Setup

### 1. Configure your environment

```bash
# Set your project
export PROJECT_ID="test-project-0728-467323"
gcloud config set project $PROJECT_ID

# Set your Google Cloud API key (required for Gemini 3)
export GOOGLE_CLOUD_API_KEY="your-api-key-here"

# Authenticate
gcloud auth application-default login
```

### 2. Create BigQuery and GCS resources

```bash
# Run the setup script
python contributing/samples/long_running_task/setup.py

# To verify setup
python contributing/samples/long_running_task/setup.py --verify

# To clean up resources
python contributing/samples/long_running_task/setup.py --cleanup
```

### 3. Run the demo

```bash
adk web contributing/samples/long_running_task
```

## Demo Scenarios

### Scenario 1: Long-running table scan

```
User: Scan the bigquery-public-data.samples.shakespeare table

Agent: [Calls simulate_long_running_scan]
       [Checkpoint written at async boundary]
       [Scan completes after ~5-10 seconds]
       The scan found 164,656 rows with the following findings:
       - Found 5 instances of 'to be or not to be'
       - Most common word: 'the' (27,801 occurrences)
       - Unique words: 29,066
```

### Scenario 2: Multi-stage pipeline

```
User: Run a pipeline from source_table to dest_table with transformations:
      filter, aggregate, join

Agent: [Calls run_data_pipeline]
       [Checkpoint written at each stage boundary]
       Pipeline completed successfully:
       - Stage 1 (filter): 45,000 rows processed
       - Stage 2 (aggregate): 32,000 rows processed
       - Stage 3 (join): 28,000 rows processed
```

### Scenario 3: Failure recovery

1. Start a long-running scan
2. Kill the process mid-execution
3. Restart and resume with the invocation_id
4. Agent continues from the last checkpoint

## Architecture

```
                    +-----------------+
                    |     Agent       |
                    |  (LlmAgent)     |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |     Runner      |
                    | (with durability)|
                    +--------+--------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
    +--------------+                 +----------------+
    |   BigQuery   |                 |      GCS       |
    |  (metadata)  |                 | (state blobs)  |
    +--------------+                 +----------------+
    | - sessions   |                 | - checkpoints/ |
    | - checkpoints|                 |   {session_id}/|
    +--------------+                 +----------------+
```

## Configuration

The agent is configured in `agent.py`:

```python
app = App(
    name="durable_session_demo",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
    durable_session_config=DurableSessionConfig(
        is_durable=True,
        checkpoint_policy="async_boundary",
        checkpoint_store=BigQueryCheckpointStore(
            project=PROJECT_ID,
            dataset=DATASET,
            gcs_bucket=GCS_BUCKET,
        ),
        lease_timeout_seconds=300,
    ),
)
```

### Checkpoint Policies

- `async_boundary`: Checkpoint when hitting async/long-running operations
- `every_turn`: Checkpoint after every agent turn
- `manual`: Only checkpoint when explicitly requested

## Monitoring

### View sessions

```sql
SELECT * FROM `test-project-0728-467323.adk_metadata.sessions`
ORDER BY updated_at DESC
LIMIT 10;
```

### View checkpoints

```sql
SELECT * FROM `test-project-0728-467323.adk_metadata.checkpoints`
ORDER BY created_at DESC
LIMIT 10;
```

### List checkpoint blobs

```bash
gsutil ls -l gs://test-project-0728-467323-adk-checkpoints/checkpoints/
```

## Cleanup

To remove all resources created by this demo:

```bash
python contributing/samples/long_running_task/setup.py --cleanup
```
