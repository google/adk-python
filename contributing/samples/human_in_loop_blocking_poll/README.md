<!--
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Human-in-the-Loop: Backend Blocking Poll Pattern

## Overview

This example demonstrates the **backend blocking poll pattern** for human-in-the-loop approval workflows. Unlike the webhook/callback pattern (`LongRunningFunctionTool`), this pattern polls an external approval API internally until a decision is made.

### How It Works

1. Agent calls approval tool **once**
2. Tool creates approval ticket via external API
3. **Tool polls API internally** every N seconds (invisible to agent)
4. Tool returns final decision to agent when ready (or timeout)

### Key Benefits

- ✅ **Simpler integration**: No `FunctionResponse` injection needed
- ✅ **Seamless UX**: Agent waits automatically, no manual "continue" clicks
- ✅ **Fewer LLM API calls**: 1 inference vs. 15+ for agent-level polling
- ✅ **Works with poll-only systems**: Jira, ServiceNow, email approvals, dashboards

---

## When to Use This Pattern vs. LongRunningFunctionTool

| Scenario | Backend Blocking Poll | LongRunningFunctionTool |
|----------|----------------------|-------------------------|
| **Poll-only system** (Jira, ServiceNow, custom dashboards) | ✅ Perfect fit | ❌ Complex application logic |
| **Webhook-capable system** (GitHub, Slack, webhooks) | ❌ Overkill | ✅ Preferred |
| **Email approval workflows** (user clicks link, poll for response) | ✅ Simple | ❌ Complex |
| **User needs real-time updates** (e.g., "Still waiting...") | ❌ Blocks silently | ✅ Can show progress |
| **Single decision** (<10 minutes) | ✅ Ideal | ⚠️  Overkill |
| **Multi-step approval** (chain of approvers) | ⚠️  Works but may timeout | ✅ Can handle state transitions |
| **High concurrency** (many simultaneous approvals) | ✅ Use async version | ✅ Both work well |

### Quick Decision Guide

**Use Backend Blocking Poll when:**
- External system doesn't support webhooks
- Simple approval workflow (single decision)
- Prefer simple application code (no manual `FunctionResponse` management)
- Approval typically completes in <10 minutes

**Use LongRunningFunctionTool when:**
- External system supports webhooks or callbacks
- Need to show progress updates to user during waiting
- Multi-step approval workflows with state transitions
- Very long-duration approvals (>10 minutes)

---

## Files in This Example

### Core Patterns

1. **`blocking_poll_approval_example.py`** - Synchronous version
   - Uses `requests` and `time.sleep()`
   - Simple, straightforward implementation
   - Good for standalone agents or low-concurrency scenarios

2. **`blocking_poll_approval_example_async.py`** - Asynchronous version
   - Uses `aiohttp` and `asyncio.sleep()`
   - Non-blocking I/O for better concurrency
   - **Recommended for production** multi-agent systems

### Testing Infrastructure

3. **`mock_approval_api.py`** - Mock approval API server
   - FastAPI-based test server
   - HTML dashboard for manual testing
   - Simulates external approval system

4. **`test_blocking_poll.py`** - Automated test script
   - Tests sync version with simulated approver
   - Validates pattern behavior

---

## Setup

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install google-adk aiohttp fastapi uvicorn requests
```

### Running the Mock Approval API

The mock API simulates an external approval system for testing.

```bash
# Start mock approval API server
python mock_approval_api.py

# Server starts at http://localhost:8003
# Dashboard: http://localhost:8003/
```

The dashboard provides a simple UI to manually approve/reject tickets during testing.

---

## Usage

### Synchronous Version

```python
from blocking_poll_approval_example import approval_agent, request_approval_blocking

# Option 1: Use the tool function directly
result = request_approval_blocking(
    proposal="Deploy version 2.0 to production",
    context={"priority": "high", "requester": "john.doe"}
)
print(result)
# ✅ APPROVED by jane.smith
# Reason: Tests passing, staging validated
# Next Action: Proceed with deployment

# Option 2: Use via ADK AgentRunner
from google.adk import AgentRunner

agent_runner = AgentRunner(approval_agent)
result = agent_runner.run(
    user_id="user123",
    new_message="Please get approval for deploying to production"
)
```

### Asynchronous Version (Recommended for Production)

```python
from blocking_poll_approval_example_async import (
    approval_agent_async,
    request_approval_blocking_async
)
import asyncio

# Option 1: Use the tool function directly
async def main():
    result = await request_approval_blocking_async(
        proposal="Deploy version 2.0 to production",
        context={"priority": "high", "requester": "john.doe"}
    )
    print(result)

asyncio.run(main())

# Option 2: Use via ADK AgentRunner (async)
from google.adk import AgentRunner

agent_runner = AgentRunner(approval_agent_async)
result = await agent_runner.run_async(
    user_id="user123",
    new_message="Please get approval for deploying to production"
)
```

---

## Testing

### Automated Test

```bash
# 1. Start mock approval API
python mock_approval_api.py

# 2. In another terminal, run test
python test_blocking_poll.py
```

Expected output:
```
✅ Approval API is running

Testing Backend Blocking Poll Pattern
[Test] Creating approval ticket...
✅ Ticket created: APR-XXXXXXXX
[Test] Starting simulated approver (will approve in 5 seconds)...
[Test] Calling request_approval_blocking (will block until approval)...
[Test] Blocking poll completed in 5.1 seconds

[Result]:
✅ APPROVED by automated_test
Reason: Auto-approved for testing
Next Action: Proceed with test

✅ TEST PASSED: Pattern works correctly!
```

### Manual Test

1. Start mock approval API: `python mock_approval_api.py`
2. Run sync or async example: `python blocking_poll_approval_example.py`
3. Open dashboard: http://localhost:8003/
4. Approve/reject pending ticket in the dashboard
5. Observe tool returns decision when ticket is decided

---

## Configuration

All configuration via environment variables:

```bash
# Approval API settings
export APPROVAL_API_URL="http://localhost:8003"  # API endpoint
export APPROVAL_POLL_INTERVAL="30"               # Seconds between polls
export APPROVAL_TIMEOUT="600"                    # Max wait time (10 minutes)

# Optional authentication
export APPROVAL_API_TOKEN="your-api-token-here"  # Bearer token for API auth

# Run example
python blocking_poll_approval_example_async.py
```

---

## Production Considerations

### Concurrency

Use the **async version** (`blocking_poll_approval_example_async.py`) for production deployments with multiple concurrent approvals. The sync version is suitable for standalone agents or low-volume scenarios (<10 concurrent approvals).

### Security

**Authentication**: Set `APPROVAL_API_TOKEN` environment variable for Bearer token authentication.

**HTTPS**: Configure `APPROVAL_API_URL` to use HTTPS in production (`https://approvals.yourcompany.com`).

**Input Validation**: Proposals are limited to 10,000 characters and cannot be empty.

### Configuration

**Timeouts**: Adjust `APPROVAL_TIMEOUT` based on your workflow:
- Fast approvals (manager): 300s (5 minutes)
- Standard approvals: 600s (10 minutes) - default
- Complex approvals (committee): 1800s (30 minutes)

**Poll Interval**: Balance responsiveness vs. API load with `APPROVAL_POLL_INTERVAL` (default: 30s).

### Monitoring

Monitor these key metrics:
- Approval creation success rate
- Average approval duration
- Timeout rate
- API error rate

The pattern includes structured logging for all approval lifecycle events.

---

## Performance Metrics (Production Validation)

This pattern has been validated in production multi-agent workflows:

| Metric | Agent-Level Polling (Anti-Pattern) | Backend Blocking Poll |
|--------|-------------------------------------|----------------------|
| **LLM API calls** | 15+ per approval | **1 per approval** |
| **Manual user clicks** | 20+ "continue" clicks | **0 clicks** |
| **Application complexity** | High (manual FunctionResponse injection) | **Low (tool handles everything)** |
| **API call reduction** | Baseline | **93% reduction** |
| **UX friction** | High (manual polling) | **Minimal (seamless)** |

**Production Workflow Example**:
- Multi-agent RFQ approval system
- 10-minute average approval duration
- Handled gracefully with no manual intervention
- 93% reduction in LLM API calls vs. agent-level polling

---

## Comparison with ADK's LongRunningFunctionTool Pattern

### LongRunningFunctionTool Workflow

```python
# 1. Tool returns "pending" immediately
def ask_for_approval(context):
    return {"status": "pending", "ticket_id": "xxx"}

# 2. Agent acknowledges pending state
# 3. External system completes task
# 4. Application code MUST manually inject FunctionResponse:
updated_response = types.Part(
    function_response=types.FunctionResponse(
        id=original_call.id,  # Must track original call ID
        name=original_call.name,
        response={"status": "approved", ...}
    )
)
await runner.run_async(new_message=types.Content(parts=[updated_response], role="user"))
```

**Complexity**: Requires manual tracking of `FunctionCall.id` and constructing `FunctionResponse`.

### Backend Blocking Poll Workflow

```python
# 1. Agent calls tool once
result = await request_approval_blocking_async(proposal)

# 2. Tool returns final decision (or timeout)
# That's it! No manual FunctionResponse injection needed.
```

**Simplicity**: Tool handles everything internally.

---

## Troubleshooting

**"Cannot connect to approval API"**
- Ensure mock API is running: `python mock_approval_api.py`
- Verify `APPROVAL_API_URL` is correct
- Check firewall/network connectivity

**"Approval timeout"**
- Check approval dashboard at configured URL
- Increase `APPROVAL_TIMEOUT` if needed
- Verify approver has access to decision interface

**"Configuration error: APPROVAL_TIMEOUT must be greater than APPROVAL_POLL_INTERVAL"**
- Ensure `APPROVAL_TIMEOUT` > `APPROVAL_POLL_INTERVAL` (e.g., 600 > 30)

**High API call volume**
- Increase `APPROVAL_POLL_INTERVAL` to reduce polling frequency (trade-off: slower response time)

---

## Additional Resources

For questions or feedback:
- [ADK Documentation](https://google.adk.dev)
- [ADK GitHub Repository](https://github.com/google/adk-python)
- Open issues on [ADK GitHub Issues](https://github.com/google/adk-python/issues)
- Reference this pattern when discussing issues [#3184](https://github.com/google/adk-python/issues/3184) or [#1797](https://github.com/google/adk-python/issues/1797)
