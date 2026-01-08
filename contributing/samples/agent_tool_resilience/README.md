# AgentTool Resilience: Timeout, Retry, and Redirect Patterns

This sample demonstrates how to handle failures, timeouts, and partial results from downstream agents in multi-agent workflows using ADK.

## Running the Demo

```bash
adk web contributing/samples/agent_tool_resilience
```

Then in the web UI, select `agent_tool_resilience` from the dropdown and try:
1. Simple query: "What is quantum computing?"
2. Complex query: (very detailed research request) as "Research quantum computing applications in healthcare, finance, cryptography, 
          logistics, weather prediction, drug discovery, and machine learning. 
          For each domain, provide: historical context, current state-of-the-art, 
          technical challenges, recent breakthroughs in 2024, comparison with classical 
          approaches, economic impact, and future roadmap for the next 10 years."
3. Timeout scenario: Set timeout to 5 seconds in `agent.py` and use a complex query

## Features Demonstrated

- **Timeout Protection**: Custom `TimeoutAgentTool` wrapper adds timeout handling to sub-agents
- **Automatic Retry**: `ReflectAndRetryToolPlugin` handles retries with structured guidance
- **Dynamic Fallback**: Coordinator agent routes to alternative agents when primary fails
- **Error Recovery**: Specialized agent provides user-friendly error analysis

## Expected Behavior

1. **Normal Operation**: Primary agent handles the query successfully
2. **Timeout Scenario**: Primary times out → Fallback agent is automatically tried
3. **Failure Scenario**: Primary fails → Retry → Fallback → Error recovery agent provides guidance

## Architecture

The sample includes:
- `coordinator_agent` - Routes requests and handles errors
- `research_agent_primary` - Primary agent with timeout protection (5s)
- `research_agent_fallback` - Fallback agent with longer timeout (60s)
- `error_recovery_agent` - Analyzes failures and provides recommendations
