# Gemini Google Search Integration Guide

## Overview

This guide documents the proper implementation of Google Search using Gemini's built-in grounding capabilities for the ADK Security Agent, as specified in the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/google-search).

## Key Difference from Custom Search API

❌ **Incorrect Approach**: Using Google Custom Search API separately
✅ **Correct Approach**: Using Gemini's native Google Search grounding

## Architecture

### Core Component

**Search-Enabled Agent** (`/agents/search_enabled_agent.py`)
- Uses Vertex AI's Gemini model
- Native Google Search grounding via `Tool.from_google_search_retrieval()`
- Automatic source citations
- No separate API keys needed

## How It Works

### 1. Model Initialization with Search Grounding

```python
from vertexai.generative_models import GenerativeModel, Tool, grounding

# Configure Google Search grounding
google_search_tool = Tool.from_google_search_retrieval(
    grounding.GoogleSearchRetrieval()
)

# Create model with search capability
model = GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[google_search_tool]
)
```

### 2. Making Search-Enhanced Queries

```python
# The model automatically searches when needed
response = model.generate_content(
    "What are the latest security vulnerabilities in Kubernetes?"
)

# Response includes grounded information from search
print(response.text)  # Contains search-enhanced answer
print(response.grounding_metadata)  # Contains sources and citations
```

## Setup Instructions

### Prerequisites

1. **Vertex AI Setup**
   ```bash
   # Install required package
   pip install google-cloud-aiplatform
   
   # Authenticate
   gcloud auth application-default login
   
   # Set project
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable APIs**
   - Vertex AI API
   - Generative Language API

### Configuration

No additional API keys needed! Gemini uses your Vertex AI credentials.

```python
import vertexai

# Initialize Vertex AI
vertexai.init(
    project="your-project-id",
    location="us-central1"
)
```

## Usage Examples

### Basic Search Query

```python
from agents.search_enabled_agent import create_search_enabled_agent

# Create agent
agent = create_search_enabled_agent(
    project_id="your-project-id",
    agent_type="conversational"
)

# Query with automatic search
response = await agent.search_and_respond(
    "Find the latest GCP security best practices"
)

print(response["response"])  # Search-grounded answer
print(response["citations"])  # Source citations
```

### Security-Focused Search

```python
# Search for security information
response = await agent.search_security_topic(
    "zero trust architecture implementation",
    include_latest=True
)
```

### Conversational Search

```python
# Maintains context across searches
agent = ConversationalSearchAgent(project_id)

response1 = await agent.search_with_context("What is IAM?")
response2 = await agent.search_with_context("How do I implement it?")
# Second query understands "it" refers to IAM
```

## Agent Integration

### Query Routing

Search queries are automatically detected and routed to the search-enabled agent:

```python
# Keywords that trigger search agent:
search_indicators = [
    "search", "find", "lookup", "research",
    "what is", "how to", "latest", "recent",
    "news", "documentation", "examples"
]
```

### Example Flow

1. User: "Search for recent Kubernetes vulnerabilities"
2. System detects search intent → Routes to SearchAgent
3. SearchAgent uses Gemini with Google Search grounding
4. Gemini searches and synthesizes information
5. Response includes answer + citations

## Features

### Automatic Grounding
- Gemini automatically determines when to search
- Seamlessly blends model knowledge with search results
- No manual API calls needed

### Source Citations
- Automatic citation extraction
- Source URLs and titles included
- Confidence scores available

### Context Preservation
- Conversational memory maintained
- Previous searches inform current queries
- Session-based context

### Security Enhancement
- Security-focused system instructions
- Automatic security context for relevant queries
- Compliance and best practices emphasis

## Response Structure

```json
{
  "success": true,
  "query": "user query",
  "response": "Gemini's search-grounded response",
  "grounding_metadata": {
    "search_queries": ["actual searches performed"],
    "sources": [
      {
        "title": "Source Title",
        "uri": "https://source.url"
      }
    ],
    "confidence": 0.95
  },
  "citations": [
    "[1] Source Title - https://source.url"
  ],
  "model_used": "gemini-1.5-pro with Google Search",
  "search_performed": true
}
```

## Advantages Over Custom Search API

1. **No Additional API Keys**: Uses Vertex AI authentication
2. **Intelligent Search**: Model decides when/what to search
3. **Better Integration**: Native grounding vs separate API calls
4. **Automatic Synthesis**: Combines multiple sources seamlessly
5. **Citation Management**: Built-in source tracking
6. **Cost Efficiency**: Single API call vs multiple

## Best Practices

1. **System Instructions**: Provide clear instructions for search behavior
2. **Temperature Settings**: Lower temperature (0.7) for factual searches
3. **Safety Settings**: Configure appropriately for your use case
4. **Model Selection**: 
   - `gemini-1.5-pro`: Best quality, higher latency
   - `gemini-1.5-flash`: Faster responses, good quality

## Monitoring and Debugging

### Check Grounding Metadata

```python
response = model.generate_content(query)

# Inspect what searches were performed
if hasattr(response, 'grounding_metadata'):
    print("Searches:", response.grounding_metadata.search_queries)
    print("Sources:", response.grounding_metadata.grounding_chunks)
```

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Limitations

1. **Region Availability**: Google Search grounding may not be available in all regions
2. **Rate Limits**: Subject to Vertex AI quotas
3. **Content Filtering**: Some content may be filtered by safety settings
4. **Search Scope**: Searches the public web (no private data)

## Testing

```python
# Test search functionality
async def test_search():
    agent = create_search_enabled_agent("project-id")
    
    test_queries = [
        "Latest CVE vulnerabilities",
        "GCP IAM best practices 2024",
        "How to implement zero trust"
    ]
    
    for query in test_queries:
        response = await agent.search_and_respond(query)
        assert response["success"]
        assert response["search_performed"]
        print(f"✅ {query}: {len(response['response'])} chars")
```

## Migration from Custom Search

If you previously used Custom Search API:

1. Remove Custom Search API dependencies
2. Remove API key management
3. Replace with `search_enabled_agent.py`
4. Update routing to use new agent
5. No changes needed to frontend

## Support and Documentation

- [Gemini API Docs](https://ai.google.dev/gemini-api/docs/google-search)
- [Vertex AI Grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)
- [Tool Use in Gemini](https://ai.google.dev/gemini-api/docs/function-calling)

## Summary

The correct implementation uses Gemini's native Google Search grounding, not a separate Custom Search API. This approach is:
- Simpler (no extra API keys)
- More intelligent (model-driven search)
- Better integrated (native grounding)
- More cost-effective (single API)