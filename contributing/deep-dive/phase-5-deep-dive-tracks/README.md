# Phase 5: Deep Dive Tracks

**Duration:** Week 2-3
**Outcome:** Specialization in one area of ADK

---

## Overview

Choose ONE track to deep dive into. Each track provides specialized knowledge that complements the core understanding you've built.

| Track | Focus Area | Key Modules |
|-------|-----------|-------------|
| **A. LLM Integration** | Model adapters, flows | `models/`, `flows/` |
| **B. Tool Ecosystem** | External integrations | `tools/openapi_tool/`, `tools/mcp_tool/` |
| **C. Evaluation** | Testing agents | `evaluation/` |
| **D. Deployment** | Production readiness | `cli/`, `apps/` |

---

## Track A: LLM Integration

**Goal:** Understand how ADK normalizes different LLM providers.

### Key Files to Study

```
src/google/adk/models/
├── base_llm.py              # Abstract LLM interface
├── llm_request.py           # Request normalization
├── llm_response.py          # Response normalization
├── google_llm.py            # Gemini integration
├── anthropic_llm.py         # Claude integration
├── lite_llm.py              # Multi-provider (OpenAI, etc.)
├── registry.py              # Model discovery
└── gemini_llm_connection.py # Connection management
```

### Core Concepts

#### 1. Request/Response Normalization

ADK uses a normalized format for all LLM interactions:

```python
# llm_request.py
class LlmRequest(BaseModel):
    """Normalized request format for all LLMs."""
    contents: list[types.Content]      # Conversation history
    config: types.GenerateContentConfig # Generation settings
    tools: list[types.Tool]            # Tool declarations
    system_instruction: Optional[types.Content]
```

```python
# llm_response.py
class LlmResponse(BaseModel):
    """Normalized response format from all LLMs."""
    content: Optional[types.Content]   # Model's response
    partial: Optional[bool]            # For streaming
    usage_metadata: Optional[UsageMetadata]  # Token counts
    # Audio/video fields for multimodal
    input_transcription: Optional[Transcription]
    output_transcription: Optional[Transcription]
```

#### 2. Model Registration

```python
# registry.py
class LLMRegistry:
    """Registry for LLM implementations."""

    @classmethod
    def register(cls, model_prefix: str, llm_class: Type[BaseLlm]):
        """Register an LLM class for a model prefix."""

    @classmethod
    def get_llm(cls, model: str) -> BaseLlm:
        """Get LLM instance for a model name."""

# Usage
from google.adk.models import LLMRegistry

# Get appropriate LLM for model name
llm = LLMRegistry.get_llm("gemini-2.0-flash")  # Returns GoogleLLM
llm = LLMRegistry.get_llm("claude-3-opus")      # Returns AnthropicLLM
```

#### 3. Creating a Custom LLM Adapter

```python
# File: my_custom_llm.py
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from typing import AsyncGenerator


class MyCustomLLM(BaseLlm):
    """Custom LLM integration."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def supported_models(self) -> list[str]:
        return ["my-model-v1", "my-model-v2"]

    async def generate_content_async(
        self,
        request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        # Convert ADK format to your API format
        my_api_request = self._convert_request(request)

        # Call your API
        if stream:
            async for chunk in self._stream_call(my_api_request):
                yield self._convert_response(chunk, partial=True)
        else:
            response = await self._single_call(my_api_request)
            yield self._convert_response(response, partial=False)

    def _convert_request(self, request: LlmRequest) -> dict:
        """Convert ADK request to your API format."""
        # Map contents, tools, config to your API
        ...

    def _convert_response(self, response: dict, partial: bool) -> LlmResponse:
        """Convert your API response to ADK format."""
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=response["text"])],
            ),
            partial=partial,
        )
```

### Study the Flows

**File:** `src/google/adk/flows/llm_flows/`

Flows orchestrate LLM calls with tool execution:

```python
# auto_flow.py - Automatic tool calling loop
# single_flow.py - Single LLM call
# base_llm_flow.py - Base flow interface
```

---

## Track B: Tool Ecosystem

**Goal:** Master external integrations through tools.

### Key Files to Study

```
src/google/adk/tools/
├── openapi_tool/           # OpenAPI/REST integration
│   ├── openapi_tool.py
│   ├── openapi_spec_parser.py
│   └── operation_mapper.py
├── mcp_tool/               # Model Context Protocol
│   ├── mcp_toolset.py
│   └── mcp_session.py
├── google_api_tool/        # Google APIs
├── retrieval/              # RAG tools
│   ├── llama_index_tool.py
│   └── vertex_ai_rag_tool.py
└── database_tools/         # Database access
```

### OpenAPI Tool Deep Dive

Convert any OpenAPI spec into callable tools:

```python
from google.adk.tools import OpenAPIToolset

# Load from OpenAPI spec
toolset = OpenAPIToolset.from_file("api_spec.yaml")

# Or from URL
toolset = OpenAPIToolset.from_url("https://api.example.com/openapi.json")

# Use with agent
agent = LlmAgent(
    name="api_agent",
    model="gemini-2.0-flash",
    tools=[toolset],
)
```

#### How It Works

```python
# openapi_spec_parser.py
class OpenAPISpecParser:
    """Parses OpenAPI specs into tool definitions."""

    def parse(self, spec: dict) -> list[OperationInfo]:
        """Extract operations from spec."""
        operations = []
        for path, methods in spec["paths"].items():
            for method, details in methods.items():
                operations.append(OperationInfo(
                    path=path,
                    method=method,
                    parameters=details.get("parameters", []),
                    request_body=details.get("requestBody"),
                    ...
                ))
        return operations
```

### MCP (Model Context Protocol) Integration

Connect to MCP servers for extended capabilities:

```python
from google.adk.tools import MCPToolset

# Connect to MCP server
toolset = await MCPToolset.from_server(
    server_params={
        "command": "npx",
        "args": ["-y", "@mcp/server-github"],
        "env": {"GITHUB_TOKEN": "..."},
    }
)

# Tools are automatically discovered
agent = LlmAgent(
    name="github_agent",
    model="gemini-2.0-flash",
    tools=[toolset],
)
```

### Creating a Custom Toolset

For grouping related tools:

```python
from google.adk.tools import BaseToolset, BaseTool
from google.adk.agents import ReadonlyContext


class WeatherToolset(BaseToolset):
    """Collection of weather-related tools."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_tools_with_prefix(
        self, ctx: ReadonlyContext
    ) -> list[BaseTool]:
        """Return all tools in this toolset."""
        return [
            FunctionTool(func=self._get_current_weather),
            FunctionTool(func=self._get_forecast),
            FunctionTool(func=self._get_alerts),
        ]

    async def _get_current_weather(self, city: str) -> dict:
        """Get current weather for a city."""
        # Call weather API
        ...

    async def _get_forecast(self, city: str, days: int = 7) -> dict:
        """Get weather forecast."""
        ...

    async def _get_alerts(self, region: str) -> list[dict]:
        """Get weather alerts."""
        ...
```

---

## Track C: Evaluation

**Goal:** Learn to systematically test and evaluate agents.

### Key Files to Study

```
src/google/adk/evaluation/
├── agent_evaluator.py       # Main evaluator
├── eval_case.py             # Test case definition
├── eval_set.py              # Collection of cases
├── eval_metrics.py          # Metric definitions
├── eval_result.py           # Result aggregation
├── llm_as_judge.py          # LLM-based evaluation
├── trajectory_evaluator.py  # Tool call evaluation
├── rubric_based_evaluator.py # Custom rubrics
└── evaluation_generator.py  # Auto-generate cases
```

### Creating Evaluation Cases

```python
# eval_cases.json
{
  "eval_set_id": "weather_agent_tests",
  "eval_cases": [
    {
      "case_id": "simple_weather_query",
      "conversation_turns": [
        {
          "user_query": "What's the weather in NYC?",
          "expected_tool_calls": [
            {
              "tool_name": "get_weather",
              "tool_args": {"city": "NYC"}
            }
          ],
          "expected_response_patterns": [
            "weather",
            "NYC|New York"
          ]
        }
      ]
    },
    {
      "case_id": "multi_step_query",
      "conversation_turns": [
        {
          "user_query": "Compare weather in NYC and LA",
          "expected_tool_calls": [
            {"tool_name": "get_weather", "tool_args": {"city": "NYC"}},
            {"tool_name": "get_weather", "tool_args": {"city": "LA"}}
          ]
        }
      ]
    }
  ]
}
```

### Running Evaluations

```python
from google.adk.evaluation import AgentEvaluator, EvalSet

# Load eval set
eval_set = EvalSet.from_file("eval_cases.json")

# Create evaluator
evaluator = AgentEvaluator(
    agent=my_agent,
    eval_set=eval_set,
)

# Run evaluation
results = await evaluator.run()

# Analyze results
print(f"Overall Score: {results.overall_score}")
print(f"Tool Trajectory Score: {results.tool_trajectory_avg_score}")
print(f"Response Match Score: {results.response_match_score}")
```

### LLM-as-Judge Evaluation

Use an LLM to evaluate response quality:

```python
from google.adk.evaluation import LlmAsJudge

judge = LlmAsJudge(
    model="gemini-2.0-flash",
    rubric="""
    Evaluate the response on these criteria:
    1. Accuracy (1-5): Is the information correct?
    2. Completeness (1-5): Does it answer the full question?
    3. Clarity (1-5): Is it easy to understand?

    Output JSON: {"accuracy": X, "completeness": X, "clarity": X, "explanation": "..."}
    """
)

score = await judge.evaluate(
    query="What's the weather in NYC?",
    response="It's currently 72°F and sunny in New York City.",
    reference="The weather in NYC is 72°F with clear skies.",
)
```

### CLI Evaluation

```bash
# Run evaluation from command line
adk eval ./my_agent --eval-set ./eval_cases.json

# With specific metrics
adk eval ./my_agent --eval-set ./eval_cases.json --metrics tool_trajectory,response_match
```

---

## Track D: Deployment

**Goal:** Master production deployment patterns.

### Key Files to Study

```
src/google/adk/
├── cli/
│   ├── cli.py              # Main CLI entry
│   ├── cli_deploy.py       # Deployment commands
│   ├── fast_api.py         # FastAPI generation
│   └── adk_web_server.py   # Dev server
└── apps/
    ├── app.py              # App configuration
    └── compaction.py       # Event compaction
```

### Creating a Production App

```python
# app.py
from google.adk.apps import App, ResumabilityConfig, EventsCompactionConfig
from google.adk.agents import LlmAgent
from google.adk.plugins import LoggingPlugin

agent = LlmAgent(
    name="production_agent",
    model="gemini-2.0-flash",
    instruction="You are a production assistant.",
)

app = App(
    name="my_production_app",
    root_agent=agent,

    # Enable session resumability
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),

    # Event compaction for long sessions
    events_compaction_config=EventsCompactionConfig(
        max_events=100,
        summarization_threshold=80,
    ),

    # Production plugins
    plugins=[
        LoggingPlugin(),
        # Add your monitoring plugins
    ],
)
```

### FastAPI Integration

```python
# server.py
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService

# Create FastAPI app from agent directory
app = get_fast_api_app(
    agent_dir="./agents",
    session_db_url="postgresql://...",
)

# Add custom endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return {"requests": 1000, "errors": 5}

# Run with uvicorn
# uvicorn server:app --host 0.0.0.0 --port 8080
```

### Deployment Options

#### Cloud Run

```bash
# Deploy to Cloud Run
adk deploy cloud_run \
    --agent-dir ./agents \
    --project my-gcp-project \
    --region us-central1
```

#### Vertex AI Agent Engine

```bash
# Deploy to Vertex AI
adk deploy vertex_ai \
    --agent-dir ./agents \
    --project my-gcp-project \
    --staging-bucket gs://my-bucket
```

#### Custom Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Production Checklist

```markdown
## Pre-Deployment Checklist

### Security
- [ ] API keys in environment variables (not code)
- [ ] Input validation on all tools
- [ ] Rate limiting configured
- [ ] Authentication enabled

### Reliability
- [ ] Session persistence configured
- [ ] Error handling in all tools
- [ ] Retry logic for external APIs
- [ ] Health check endpoint

### Monitoring
- [ ] Logging plugin enabled
- [ ] Metrics collection configured
- [ ] Alerting set up
- [ ] Tracing enabled (OpenTelemetry)

### Performance
- [ ] Event compaction for long sessions
- [ ] Context caching enabled
- [ ] Connection pooling for databases
- [ ] Async operations used throughout
```

---

## Track Selection Guide

| If you want to... | Choose Track |
|-------------------|--------------|
| Add support for new LLM providers | A. LLM Integration |
| Integrate with external APIs/services | B. Tool Ecosystem |
| Build a testing pipeline for agents | C. Evaluation |
| Deploy agents to production | D. Deployment |

---

**Next:** [Phase 6: Contributing](../phase-6-contributing/README.md)
