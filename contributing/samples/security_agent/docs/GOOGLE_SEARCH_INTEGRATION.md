# Google Web Search Integration Guide

## Overview

This guide documents the Google Web Search integration for the ADK Security Agent, enabling real-time web search capabilities to enhance agent responses with current information.

## Architecture

### Components

1. **Search Service** (`/backend/api/search.py`)
   - Core service handling Google Custom Search API
   - Rate limiting and caching
   - Mock data for development

2. **Search Tools** (`/tools/api_tools/google_search_tools.py`)
   - ADK-compliant tool functions
   - Security-focused search capabilities
   - Context analysis

3. **Data Models** (`/backend/models/search_models.py`)
   - Pydantic models for validation
   - Request/response structures
   - Analytics models

4. **Agent Integration** (`/backend/api/agent_llm.py`)
   - SearchAgent routing
   - Query intent detection
   - Response formatting

## Setup Instructions

### 1. Google API Configuration

#### Get Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable "Custom Search API"
4. Create credentials (API Key)
5. Copy the API key

#### Create Custom Search Engine
1. Go to [Google Custom Search Engine](https://cse.google.com/)
2. Click "Add" to create a new search engine
3. Configure:
   - Sites to search: Select "Search the entire web"
   - Enable "SafeSearch"
   - Get the Search Engine ID (cx)

### 2. Environment Configuration

Add to your `.env` file:

```bash
# Google Search Configuration
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_search_engine_id_here

# Optional Settings
SEARCH_RATE_LIMIT_PER_MINUTE=100
SEARCH_CACHE_TTL_SECONDS=3600
```

### 3. Backend Integration

The search service is automatically registered when the backend starts:

```python
# In main.py or app initialization
from backend.api.search import router as search_router
app.include_router(search_router)
```

## Usage

### Tool Functions

#### Basic Web Search
```python
from tools.api_tools.google_search_tools import search_web

result = search_web(
    query="GCP security best practices",
    max_results=10,
    safe_search=True,
    session_id="session123"
)
```

#### Security-Focused Search
```python
from tools.api_tools.google_search_tools import search_security_topics

result = search_security_topics(
    query="authentication vulnerabilities",
    session_id="session123"
)
# Returns results with security analysis and recommendations
```

#### Context-Aware Search
```python
from tools.api_tools.google_search_tools import get_search_context

context = get_search_context(
    session_id="session123",
    query="current topic",
    include_history=True
)
# Returns contextual suggestions based on conversation
```

### API Endpoints

#### POST /api/v1/search/web
Perform web search:
```json
{
  "query": "cloud security",
  "session_id": "session123",
  "user_id": "user123",
  "max_results": 10,
  "safe_search": true
}
```

Response:
```json
{
  "success": true,
  "query": "cloud security",
  "results": [
    {
      "title": "Cloud Security Best Practices",
      "url": "https://example.com",
      "snippet": "Comprehensive guide...",
      "display_url": "example.com",
      "relevance_score": 0.95
    }
  ],
  "total_results": 1000,
  "search_time_ms": 250,
  "llm_summary": "Found comprehensive guides on cloud security...",
  "suggested_refinements": ["cloud security gcp", "cloud security aws"],
  "security_context": {
    "is_security_related": true,
    "key_concerns": ["data protection", "access control"]
  }
}
```

#### POST /api/v1/search/context
Get search context and suggestions:
```json
{
  "session_id": "session123",
  "query": "authentication",
  "include_history": true,
  "analyze_security": true
}
```

#### GET /api/v1/search/history/{session_id}
Retrieve search history for a session

#### GET /api/v1/search/config
Get current search configuration status

## Agent Integration

### Query Routing

The agent automatically detects search intent based on keywords:

- **Search indicators**: "search", "find", "lookup", "research", "what is", "how to"
- **Information queries**: "latest", "recent", "news", "documentation"
- **Examples**: "examples of", "show me", "list"

### Example Interactions

```
User: "Search for GCP IAM best practices"
Agent: Routes to SearchAgent → Performs web search → Returns formatted results

User: "What are the latest security vulnerabilities?"
Agent: Routes to SearchAgent → Security-focused search → Returns with analysis

User: "Find documentation on zero trust architecture"
Agent: Routes to SearchAgent → Documentation search → Returns relevant guides
```

## Features

### Rate Limiting
- Sliding window rate limiter
- Default: 100 requests per minute per user
- HTTP 429 response when exceeded
- Configurable via environment variable

### Caching
- In-memory cache for identical queries
- Default TTL: 1 hour
- Reduces API calls and improves response time
- Cache key: query + max_results + safe_search

### Mock Data
- Automatically used when API not configured
- Provides realistic responses for development
- Security-aware mock results

### Session Integration
- Search history tracked per session
- Context preservation across searches
- Conversation-aware suggestions

### Security Features
- Safe search enabled by default
- Security context analysis for relevant queries
- Recommendations for security searches
- Topic detection and categorization

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/search/health
```

### Analytics
```bash
curl -X POST http://localhost:8000/api/v1/search/analytics \
  -H "Content-Type: application/json" \
  -d '{"time_range_hours": 24}'
```

Returns:
- Total searches
- Unique queries
- Popular searches
- Rate limit status

## Testing

### Run Tests
```bash
pytest tests/test_search_integration.py -v
```

### Test Coverage
- Service functionality
- Rate limiting
- Caching
- Model validation
- Tool functions
- Agent routing
- End-to-end flow

## Troubleshooting

### Common Issues

#### "Search Service Not Available"
- Check GOOGLE_API_KEY and GOOGLE_CSE_ID are set
- Verify API is enabled in Google Cloud Console
- Check network connectivity

#### Rate Limiting
- Default limit: 100/minute
- Adjust SEARCH_RATE_LIMIT_PER_MINUTE if needed
- Implement user-specific limits if required

#### No Results
- Verify Custom Search Engine configuration
- Check if sites are properly indexed
- Try broader search terms

## Best Practices

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Rate Management**
   - Implement appropriate rate limits
   - Use caching effectively
   - Consider pagination for large result sets

3. **User Experience**
   - Provide search suggestions
   - Show relevance scores
   - Include security context when appropriate

4. **Session Management**
   - Track search history
   - Provide contextual suggestions
   - Clear old sessions periodically

## Future Enhancements

- [ ] Redis cache implementation
- [ ] Advanced query parsing
- [ ] Multi-language support
- [ ] Image search capability
- [ ] Search result clustering
- [ ] Machine learning relevance ranking
- [ ] Federated search across multiple sources

## Support

For issues or questions:
- Check logs: `backend/logs/search.log`
- Review test suite: `tests/test_search_integration.py`
- Submit issues to the ADK repository