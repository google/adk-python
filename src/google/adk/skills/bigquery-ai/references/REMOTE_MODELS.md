# Remote Models in BigQuery

Complete guide to creating and managing connections to LLMs and AI services.

## Overview

Remote models connect BigQuery to external AI services:
- Google Gemini models
- Partner models (Claude, Llama, Mistral)
- Vertex AI endpoints
- Cloud AI services

## CREATE MODEL Syntax

```sql
CREATE [ OR REPLACE ] MODEL `project.dataset.model_name`
  REMOTE WITH CONNECTION `project.region.connection_name`
  OPTIONS (
    ENDPOINT = 'model_endpoint',
    [additional_options]
  );
```

## Prerequisites

### 1. Create a Cloud Resource Connection

```sql
-- Using BigQuery UI or bq command:
bq mk --connection \
  --connection_type=CLOUD_RESOURCE \
  --location=US \
  --project_id=my_project \
  my_connection
```

Or via API:
```bash
gcloud bigquery connections create my_connection \
  --connection-type=CLOUD_RESOURCE \
  --location=US
```

### 2. Grant IAM Permissions

The connection's service account needs roles:
- `roles/aiplatform.user` - For Vertex AI models
- `roles/bigquery.connectionUser` - For BigQuery access

```bash
# Get service account
bq show --connection --location=US my_connection

# Grant permission
gcloud projects add-iam-policy-binding my_project \
  --member="serviceAccount:bqcx-xxx@gcp-sa-bigquery-condel.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

## Google Models

### Gemini Models

```sql
-- Gemini 2.0 Flash (recommended for speed)
CREATE OR REPLACE MODEL `project.dataset.gemini_flash`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'gemini-2.0-flash');

-- Gemini 1.5 Pro (higher quality)
CREATE OR REPLACE MODEL `project.dataset.gemini_pro`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'gemini-1.5-pro');

-- Gemini 1.5 Flash
CREATE OR REPLACE MODEL `project.dataset.gemini_15_flash`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'gemini-1.5-flash');
```

### Embedding Models

```sql
-- Text Embedding (recommended)
CREATE OR REPLACE MODEL `project.dataset.text_embedding`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');

-- Previous generation
CREATE OR REPLACE MODEL `project.dataset.text_embedding_004`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-004');

-- Multilingual
CREATE OR REPLACE MODEL `project.dataset.multilingual_embedding`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-multilingual-embedding-002');

-- Multimodal (text + images)
CREATE OR REPLACE MODEL `project.dataset.multimodal_embedding`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'multimodalembedding@001');
```

## Partner Models

### Anthropic Claude

```sql
-- Claude 3.5 Sonnet
CREATE OR REPLACE MODEL `project.dataset.claude_sonnet`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'claude-3-5-sonnet@20241022'
  );

-- Claude 3 Opus (highest quality)
CREATE OR REPLACE MODEL `project.dataset.claude_opus`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'claude-3-opus@20240229'
  );

-- Claude 3 Haiku (fast)
CREATE OR REPLACE MODEL `project.dataset.claude_haiku`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'claude-3-haiku@20240307'
  );
```

### Meta Llama

```sql
-- Llama 3.1 405B
CREATE OR REPLACE MODEL `project.dataset.llama_405b`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'llama-3.1-405b-instruct-maas'
  );

-- Llama 3.1 70B
CREATE OR REPLACE MODEL `project.dataset.llama_70b`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'llama-3.1-70b-instruct-maas'
  );

-- Llama 3.2 90B Vision
CREATE OR REPLACE MODEL `project.dataset.llama_vision`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'llama-3.2-90b-vision-instruct-maas'
  );
```

### Mistral AI

```sql
-- Mistral Large
CREATE OR REPLACE MODEL `project.dataset.mistral_large`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'mistral-large@2411'
  );

-- Mistral Nemo
CREATE OR REPLACE MODEL `project.dataset.mistral_nemo`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'mistral-nemo@2407'
  );

-- Codestral (code generation)
CREATE OR REPLACE MODEL `project.dataset.codestral`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'codestral@2405'
  );
```

## Custom Vertex AI Endpoints

Connect to your own deployed models:

```sql
CREATE OR REPLACE MODEL `project.dataset.custom_model`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890'
  );
```

## Cloud AI Services

### Translation

```sql
CREATE OR REPLACE MODEL `project.dataset.translate`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_translate_v3'
  );

-- Usage
SELECT *
FROM ML.TRANSLATE(
  MODEL `project.dataset.translate`,
  (SELECT text AS text_to_translate FROM documents),
  STRUCT('es' AS target_language_code)
);
```

### Natural Language

```sql
CREATE OR REPLACE MODEL `project.dataset.nlp`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_natural_language_v1'
  );

-- Usage (sentiment, entities, syntax)
SELECT *
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp`,
  (SELECT text AS text_content FROM reviews),
  STRUCT('ANALYZE_SENTIMENT' AS nlp_task)
);
```

### Vision

```sql
CREATE OR REPLACE MODEL `project.dataset.vision`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_vision_v1'
  );

-- Usage
SELECT *
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.vision`,
  (SELECT uri FROM images),
  STRUCT(['LABEL_DETECTION', 'TEXT_DETECTION'] AS vision_features)
);
```

### Document AI

```sql
CREATE OR REPLACE MODEL `project.dataset.document_ai`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_document_v1',
    DOCUMENT_PROCESSOR = 'projects/my-project/locations/us/processors/abc123'
  );

-- Usage
SELECT *
FROM ML.PROCESS_DOCUMENT(
  MODEL `project.dataset.document_ai`,
  (SELECT uri FROM pdfs)
);
```

### Speech-to-Text

```sql
CREATE OR REPLACE MODEL `project.dataset.speech`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_speech_v2'
  );

-- Usage
SELECT *
FROM ML.TRANSCRIBE(
  MODEL `project.dataset.speech`,
  (SELECT uri FROM audio_files),
  STRUCT('en-US' AS language_code)
);
```

## Model Management

### List Models

```sql
SELECT
  model_name,
  model_type,
  creation_time,
  training_runs[SAFE_OFFSET(0)].training_options.model_type AS model_subtype
FROM `project.dataset.INFORMATION_SCHEMA.MODELS`;
```

### Get Model Details

```sql
SELECT *
FROM ML.MODEL_INFO(MODEL `project.dataset.my_model`);
```

### Drop Model

```sql
DROP MODEL IF EXISTS `project.dataset.my_model`;
```

## Model Comparison

| Model | Provider | Speed | Quality | Cost | Best For |
|-------|----------|-------|---------|------|----------|
| gemini-2.0-flash | Google | Fast | Good | Low | High volume |
| gemini-1.5-pro | Google | Medium | High | Medium | Complex tasks |
| claude-3.5-sonnet | Anthropic | Medium | High | Medium | Reasoning |
| claude-3-opus | Anthropic | Slow | Highest | High | Critical tasks |
| llama-3.1-405b | Meta | Slow | High | Medium | Open source |
| mistral-large | Mistral | Medium | High | Medium | EU compliance |

## Regional Availability

| Model | US | EU | Asia |
|-------|----|----|------|
| Gemini | Yes | Yes | Yes |
| Claude | Yes | Yes | Limited |
| Llama | Yes | Yes | Limited |
| Mistral | Yes | Yes | Limited |

## Best Practices

### 1. Use Connection Defaults

```sql
-- Simplest syntax with DEFAULT connection
CREATE MODEL `project.dataset.model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'gemini-2.0-flash');
```

### 2. Version Your Models

```sql
-- Include version in name
CREATE MODEL `project.dataset.gemini_20_flash_v1`
  ...
```

### 3. Create Per-Use-Case Models

```sql
-- Different models for different tasks
CREATE MODEL `project.dataset.summarizer` ...
CREATE MODEL `project.dataset.classifier` ...
CREATE MODEL `project.dataset.embedder` ...
```

### 4. Test Before Production

```sql
-- Test with small sample
SELECT * FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.new_model`,
  (SELECT prompt FROM test_prompts LIMIT 10)
);
```

## Troubleshooting

### Connection Issues

```sql
-- Verify connection exists
SELECT * FROM `project.INFORMATION_SCHEMA.CONNECTIONS`;

-- Check service account permissions
SELECT * FROM `project.INFORMATION_SCHEMA.OBJECT_PRIVILEGES`
WHERE grantee LIKE '%bqcx%';
```

### Model Not Found

- Verify endpoint spelling
- Check regional availability
- Ensure connection has proper IAM roles

### Rate Limits

- Use appropriate request_type
- Implement retry logic
- Consider dedicated capacity
