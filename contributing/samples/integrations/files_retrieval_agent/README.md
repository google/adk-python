# Files Retrieval Agent

A sample agent that demonstrates using `FilesRetrieval` with the
`gemini-embedding-2-preview` embedding model for retrieval-augmented
generation (RAG) over local files.

## What it does

This agent indexes local text files from the `data/` directory using
`FilesRetrieval` (backed by LlamaIndex's `VectorStoreIndex` and Google's
`gemini-embedding-2-preview` embedding model), then answers user questions
by retrieving relevant documents before generating a response.

## Prerequisites

- Python 3.11+
- `google-genai >= 1.64.0` (required for `gemini-embedding-2-preview`
  support via the Vertex AI `embedContent` endpoint)
- `llama-index-embeddings-google-genai >= 0.3.0`

Install dependencies:

```bash
uv sync --all-extras
```

## Authentication

Configure one of the following:

**Google AI API:**

```bash
export GOOGLE_API_KEY="your-api-key"
```

**Vertex AI:**

```bash
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

Note: `gemini-embedding-2-preview` is currently only available in
`us-central1`.

## Usage

```bash
cd contributing/samples

# Interactive CLI
adk run files_retrieval_agent

# Web UI
adk web .
```

## Example queries

- "What agent types does ADK support?"
- "How does FilesRetrieval work?"
- "What tools are available in ADK?"

## File structure

```
files_retrieval_agent/
├── __init__.py
├── agent.py           # Agent definition with FilesRetrieval tool
├── data/
│   ├── adk_overview.txt   # ADK architecture overview
│   └── tools_guide.txt    # ADK tools documentation
└── README.md
```
