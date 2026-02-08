# Milvus RAG Agent Sample

This sample demonstrates how to build a knowledge base agent using
[Milvus](https://milvus.io/) as the vector database for
retrieval-augmented generation (RAG) with Google ADK.

## Prerequisites

1. A running Milvus instance, or use Milvus Lite (local file path).
2. A Google GenAI API key for embedding generation.
3. Install dependencies:

```bash
pip install "google-adk[milvus]"
```

## Supported Milvus Backends

The integration works with all three Milvus deployment modes by changing
the `uri` (and optionally `token`) setting:

| Backend | `uri` | `token` |
|---------|-------|---------|
| **Milvus Lite** (local, no server needed) | `./milvus.db` | — |
| **Milvus Server** (self-hosted) | `http://localhost:19530` | — |
| **Zilliz Cloud** (fully managed) | `https://in01-xxx.serverless.gcp-us-west1.cloud.zilliz.com` | `your-api-key` |

> For Milvus Lite, install the extra package: `pip install "pymilvus[milvus_lite]"`.

## Setup

### Environment Variables

```bash
export GOOGLE_API_KEY="your-google-api-key"

# Pick one of the following:
export MILVUS_URI="./milvus.db"                # Milvus Lite
export MILVUS_URI="http://localhost:19530"      # Milvus Server
export MILVUS_URI="https://in01-xxx.cloud.zilliz.com"  # Zilliz Cloud

export MILVUS_COLLECTION="knowledge_base"
# Required for Zilliz Cloud only:
# export MILVUS_TOKEN="your-api-key"
```

### Data Ingestion

Before running the agent, you need to populate the Milvus collection
with your knowledge base data:

```python
from google.adk.tools.milvus.milvus_vector_store import MilvusVectorStore
from google.adk.tools.milvus.settings import MilvusToolSettings
from google.adk.tools.milvus.settings import MilvusVectorStoreSettings
from google.genai import Client

# Define your embedding function (example using Google GenAI).
genai_client = Client()

def embedding_fn(texts):
    resp = genai_client.models.embed_content(
        model="text-embedding-004", contents=texts)
    return [list(e.values) for e in resp.embeddings]

settings = MilvusToolSettings(
    vector_store_settings=MilvusVectorStoreSettings(
        uri="http://localhost:19530",
        collection_name="knowledge_base",
        dimension=768,
    ),
)

store = MilvusVectorStore(settings=settings, embedding_fn=embedding_fn)
store.setup()
store.add_contents([
    "Your document text here...",
    "Another document...",
])
```

## Run

```bash
adk run contributing/samples/milvus_rag_agent
```
