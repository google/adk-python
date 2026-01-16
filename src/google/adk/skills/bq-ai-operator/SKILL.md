---
name: bq-ai-operator
description: Execute AI/ML operations in BigQuery including text generation, embeddings, and remote model inference.
license: Apache-2.0
compatibility: BigQuery, Vertex AI, Gemini
metadata:
  author: Google Cloud
  version: "1.0"
  category: ai-operations
adk:
  config:
    timeout_seconds: 120
    max_parallel_calls: 10
  allowed_callers:
    - bigquery_agent
    - ai_agent
---

# BigQuery AI Operator Skill

Execute AI/ML operations directly in BigQuery using remote models and built-in AI functions.

## When to Use

- Generate text from data using LLMs
- Create embeddings for semantic search
- Classify or summarize text at scale
- Run inference on Vertex AI models from SQL

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_text` | Generate text using ML.GENERATE_TEXT |
| `generate_embedding` | Create vector embeddings using ML.GENERATE_EMBEDDING |
| `understand_text` | Analyze text for sentiment, entities, etc. |
| `translate_text` | Translate text between languages |
| `create_remote_model` | Create a connection to a Vertex AI model |
| `list_remote_models` | List available remote model connections |

## Quick Start

1. **Setup**: Create a remote model connection to Vertex AI
2. **Generate**: Use `generate_text` for LLM inference on data
3. **Embed**: Use `generate_embedding` for vector search

## References

- `REMOTE_MODELS.md` - How to connect to Vertex AI models
- `AI_FUNCTIONS.md` - Built-in AI/ML SQL functions
- `EMBEDDING_SEARCH.md` - Vector search patterns

## Scripts

- `setup_connection.py` - Create BigQuery-Vertex AI connection
- `batch_inference.py` - Run batch inference jobs
