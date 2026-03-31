# SidClaw Governance Agent

This sample shows how to add policy evaluation, human approval, and a
tamper-proof audit trail to ADK tool calls using
[SidClaw](https://sidclaw.com) — an open-source governance layer for AI agents.

The sample builds a customer support agent with three tools: `send_email`,
`get_customer_record`, and `lookup_order`. Before any tool executes, SidClaw
evaluates it against your org's policies. Operations flagged as high-risk
(based on `data_classification`) are held for human review in the SidClaw
dashboard before proceeding.

## Setup

Install dependencies:

```bash
uv pip install sidclaw google-adk
```

Get a free API key and agent ID from [app.sidclaw.com](https://app.sidclaw.com)
(no credit card required for the free tier — covers 5 agents).

Set environment variables:

```bash
export SIDCLAW_API_KEY=your_api_key
export SIDCLAW_AGENT_ID=customer-support-agent
```

## Running the agent

```bash
adk run contributing/samples/sidclaw_governance_agent
```

When the agent calls `send_email` or `get_customer_record`, SidClaw intercepts
the call, evaluates the configured policy, and — if the policy marks it as
`approval_required` — holds the action until a reviewer approves or denies it
from the dashboard or a connected Slack/Teams channel.

## What SidClaw adds

- **Policy evaluation** — named policies with priority ordering evaluate every
  tool call before execution. Allow, deny, or require human approval per
  operation type, data classification, or resource scope.
- **Human approval workflow** — reviewers see the agent's identity, what it
  wants to do, the full action payload, and the agent's reasoning before
  deciding.
- **Hash-chain audit trail** — every evaluation, approval, and execution is
  recorded in a cryptographically chained log. The trace is tamper-evident and
  exportable for compliance reviews (FINRA, EU AI Act, NIST AI RMF).

## Governance configuration

```python
from sidclaw.middleware.google_adk import GoogleADKGovernanceConfig

config = GoogleADKGovernanceConfig(
    data_classification={
        "send_email": "confidential",
        "get_customer_record": "confidential",
    },
    default_classification="internal",
    resource_scope="customer_support",
    wait_for_approval=True,
    approval_timeout_seconds=300.0,
)
```

`data_classification` maps tool names to sensitivity levels. Tools classified as
`confidential` are evaluated against stricter policies by default. Override this
in the SidClaw policy editor without changing agent code.

## SDK reference

- `govern_google_adk_tool(client, tool, config)` — wrap a single tool
- `govern_google_adk_tools(client, tools, config)` — wrap a list of tools
- `govern_google_adk_tool_async(client, tool, config)` — async variant

Source: [github.com/sidclawhq/python-sdk](https://github.com/sidclawhq/python-sdk)
