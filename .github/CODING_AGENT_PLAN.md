# CodingAgent - Implementation Plan & Status

This document tracks the implementation of CodingAgent, an experimental agent type that generates and executes Python code in sandboxed containers.

## Overview

CodingAgent is a ReAct-style agent that:
- Generates Python code to solve tasks using an LLM (Gemini)
- Executes code in sandboxed Docker containers
- Calls ADK tools from generated code via HTTP IPC
- Iterates until a final answer is produced

## Implementation Status

### Core Components ✅ Complete

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| CodingAgent | `src/google/adk/agents/coding_agent.py` | ✅ Complete | ~610 |
| CodingAgentConfig | `src/google/adk/agents/coding_agent_config.py` | ✅ Complete | ~225 |
| CodingAgentCodeExecutor | `src/google/adk/code_executors/coding_agent_code_executor.py` | ✅ Complete | ~505 |
| ToolCodeGenerator | `src/google/adk/code_executors/tool_code_generator.py` | ✅ Complete | ~475 |
| ToolExecutionServer | `src/google/adk/code_executors/tool_execution_server.py` | ✅ Complete | ~365 |
| AllowlistValidator | `src/google/adk/code_executors/allowlist_validator.py` | ✅ Complete | ~355 |

### Sample Agent ✅ Complete

| File | Status | Description |
|------|--------|-------------|
| `contributing/samples/coding_agent/agent.py` | ✅ Complete | Data Analysis Agent (~360 lines) |
| `contributing/samples/coding_agent/README.md` | ✅ Complete | Documentation (~290 lines) |
| `contributing/samples/coding_agent/__init__.py` | ✅ Complete | Module init |

### Unit Tests ✅ Complete

| Test File | Status | Lines |
|-----------|--------|-------|
| `tests/unittests/agents/test_coding_agent.py` | ✅ Complete | ~310 |
| `tests/unittests/code_executors/test_allowlist_validator.py` | ✅ Complete | ~320 |
| `tests/unittests/code_executors/test_tool_code_generator.py` | ✅ Complete | ~320 |

### Manual E2E Tests ✅ Passed

| Test Scenario | Status | Notes |
|--------------|--------|-------|
| Basic math query ("What is 25 * 17?") | ✅ Passed | Returns 425 |
| Data analysis (Titanic survival rate) | ✅ Passed | Returns 38.38% |
| Visualization (bar chart by class) | ✅ Passed | Chart saved to host |
| Multi-step analysis | ✅ Passed | Stats + visualization + insights |
| Tool calling via HTTP IPC | ✅ Passed | fetch_url, save_chart work |
| Error handling (pip warnings) | ✅ Passed | Ignores non-fatal stderr |
| Chart saving to host system | ✅ Passed | Saved to /tmp/adk_charts/ |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│   CodingAgent    │────▶│ Docker Container│
│                 │     │  (Gemini LLM)    │     │ (Python 3.11)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               │                         │ Executes
                               ▼                         │ generated code
                        ┌──────────────┐                 │
                        │ Tool Server  │◀────────────────┘
                        │ (HTTP IPC)   │  Tool calls via HTTP
                        └──────────────┘
```

### How Tool IPC Works

1. CodingAgent starts ToolExecutionServer on host (port 8765)
2. Code is generated with tool stubs that make HTTP POST requests
3. Container reaches host via `host.docker.internal` (macOS/Windows) or bridge gateway (Linux)
4. Tool server executes actual tool functions with proper context
5. Results returned to container via HTTP response

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Container image | `python:3.11-slim` + runtime pip | Simpler for users, no custom Dockerfile |
| Tool communication | HTTP IPC | Works across container boundary, secure |
| Import validation | Allowlist-based | Security without blocking legitimate use |
| Chart saving | `save_chart` tool | Transfers data to host filesystem |
| Error handling | Distinguish warnings from errors | pip warnings shouldn't fail execution |

## Sample Agent: Data Analyst

### Tools Available

| Tool | Description |
|------|-------------|
| `fetch_url(url)` | Fetch CSV/JSON/text from URLs |
| `get_sample_datasets()` | List available datasets (Titanic, Iris, Tips) |
| `get_current_time()` | Get current timestamp |
| `save_chart(image_data, filename)` | Save base64 chart to host |
| `list_saved_charts()` | List saved charts |

### Example Queries

1. "What is the survival rate on the Titanic?"
2. "Create a bar chart showing survival rate by passenger class"
3. "Analyze the iris dataset and create a scatter plot colored by species"
4. "Perform comprehensive analysis: stats, survival rates, visualization, insights"

## Files Changed Summary

```
 .github/CODING_AGENT_PLAN.md                       | Plan document
 contributing/samples/coding_agent/README.md        | 290 lines
 contributing/samples/coding_agent/__init__.py      | 17 lines
 contributing/samples/coding_agent/agent.py         | 360 lines
 src/google/adk/agents/__init__.py                  | +2 exports
 src/google/adk/agents/coding_agent.py              | 610 lines
 src/google/adk/agents/coding_agent_config.py       | 225 lines
 src/google/adk/code_executors/__init__.py          | +6 exports
 src/google/adk/code_executors/allowlist_validator.py    | 355 lines
 src/google/adk/code_executors/coding_agent_code_executor.py | 505 lines
 src/google/adk/code_executors/tool_code_generator.py    | 475 lines
 src/google/adk/code_executors/tool_execution_server.py  | 365 lines
 tests/unittests/agents/test_coding_agent.py        | 310 lines
 tests/unittests/code_executors/test_allowlist_validator.py | 320 lines
 tests/unittests/code_executors/test_tool_code_generator.py | 320 lines
```

**Total: ~4,200 lines of new code**

## PR Checklist

- [x] Implementation complete
- [x] Unit tests written and passing
- [x] Manual E2E tests passing
- [x] Sample agent created with README
- [x] Code follows ADK style guide (relative imports, `from __future__ import annotations`)
- [x] Marked as `@experimental`
- [ ] Run `./autoformat.sh` before PR
- [ ] Run full test suite: `pytest tests/unittests`
- [ ] Create GitHub issue (see `.github/CODING_AGENT_ISSUE.md`)
- [ ] Submit PR with testing plan

## Future Enhancements (Out of Scope)

- Stateful execution (persist variables across turns)
- Custom container images with pre-installed packages
- VertexAI code execution integration
- Support for JavaScript/TypeScript
- Streaming output during execution
