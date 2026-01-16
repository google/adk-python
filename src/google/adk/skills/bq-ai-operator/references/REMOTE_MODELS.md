# Remote Models in BigQuery

## Overview

Remote models connect BigQuery to external AI services, primarily Vertex AI, enabling SQL-based inference on LLMs and other models.

## Creating a Connection

### Step 1: Create a BigQuery Connection

```sql
-- Create connection to Vertex AI
CREATE EXTERNAL CONNECTION `project.region.vertex_ai_connection`
OPTIONS (
  type = 'CLOUD_RESOURCE'
)
```

### Step 2: Grant IAM Permissions

The connection's service account needs appropriate IAM roles:

```bash
# Get the service account email
bq show --connection project.region.vertex_ai_connection

# Grant Vertex AI User role
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/aiplatform.user"
```

### Step 3: Create the Remote Model

```sql
-- Create a remote model for Gemini
CREATE OR REPLACE MODEL `project.dataset.gemini_model`
REMOTE WITH CONNECTION `project.region.vertex_ai_connection`
OPTIONS (
  endpoint = 'gemini-1.5-flash'
)
```

## Supported Endpoints

### Gemini Models
| Endpoint | Description |
|----------|-------------|
| `gemini-1.5-pro` | Most capable, best for complex tasks |
| `gemini-1.5-flash` | Fast and efficient for most tasks |
| `gemini-1.0-pro` | Stable, production-ready |

### Embedding Models
| Endpoint | Description |
|----------|-------------|
| `text-embedding-004` | Latest text embedding model (768 dims) |
| `text-embedding-gecko@003` | Production text embeddings |
| `multimodalembedding@001` | Text + image embeddings |

### PaLM Models (Legacy)
| Endpoint | Description |
|----------|-------------|
| `text-bison@002` | Text generation |
| `textembedding-gecko@003` | Text embeddings |

## Model Configuration Options

```sql
CREATE OR REPLACE MODEL `project.dataset.custom_model`
REMOTE WITH CONNECTION `project.region.vertex_ai_connection`
OPTIONS (
  endpoint = 'gemini-1.5-flash',
  -- Optional settings
  max_output_tokens = 1024,
  temperature = 0.2,
  top_p = 0.95,
  top_k = 40
)
```

## Using Remote Models

### Text Generation
```sql
SELECT
  ml_generate_text_result['candidates'][0]['content'] AS response
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_model`,
  (SELECT 'Summarize this text: ' || content AS prompt FROM articles),
  STRUCT(
    0.2 AS temperature,
    1024 AS max_output_tokens
  )
)
```

### Embeddings
```sql
SELECT
  content,
  ml_generate_embedding_result['embeddings'][0]['values'] AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT content FROM documents),
  STRUCT(TRUE AS flatten_json_output)
)
```

## Best Practices

1. **Rate Limiting**: Use batch operations for large datasets
2. **Cost Control**: Set quotas on the Vertex AI endpoint
3. **Error Handling**: Check `ml_generate_text_status` for failures
4. **Caching**: Store embeddings to avoid recomputation
5. **Region**: Keep data and model in the same region
