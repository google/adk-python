# A2A Human-in-the-Loop Sample Agent

This sample demonstrates the **Agent-to-Agent (A2A)** architecture with **Human-in-the-Loop** workflows in the Agent Development Kit (ADK). The sample implements a reimbursement processing agent that automatically handles small expenses while requiring remote agent to process for larger amounts. The remote agent will require a human approval for large amounts, thus surface this request to local agent and human interacting with local agent can approve the request.

## Overview

The A2A Human-in-the-Loop sample consists of:

- **Root Agent** (`root_agent`): The main reimbursement agent that handles expense requests and delegates approval to remote Approval Agent for large amounts
- **Approval Agent** (`approval_agent`): A remote A2A agent that handles the human approval process via  long-running tools (which implements asynchronous approval workflows that can pause execution and wait for human input), this agent is running on a separate A2A server


## Architecture

```
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│   Human Manager │───▶│   Root Agent       │───▶│   Approval Agent │
│   (External)    │    │    (Local)         │    │  (Remote A2A)    │
│                 │    │                    │    │ (localhost:8001) │
│   Approval UI   │◀───│                    │◀───│                  │
└─────────────────┘    └────────────────────┘    └──────────────────┘
```

## Key Features

### 1. **Automated Decision Making**
- Automatically approves reimbursements under $100
- Uses business logic to determine when human intervention is required
- Provides immediate responses for simple cases

### 2. **Human-in-the-Loop Workflow**
- Seamlessly escalates high-value requests (>$100) to remote approval agent
- Remote approval agent uses long-running tools to surface approval requests back to the root agent
- Human managers interact directly with the root agent to approve/reject requests

### 3. **Long-Running Tool Integration**
- Demonstrates `LongRunningFunctionTool` for asynchronous operations
- Shows how to handle pending states and external updates
- Implements proper tool response handling for delayed approvals

### 4. **Remote A2A Agent Communication**
- The approval agent runs as a separate service that processes approval workflows
- Communicates via HTTP at `http://localhost:8001/a2a/human_in_loop`
- Surfaces approval requests back to the root agent for human interaction

## Setup and Usage

### Prerequisites

1. **Start the Remote Approval Agent server**:
   ```bash
   # Start the remote a2a server that serves the human-in-the-loop approval agent on port 8001
   adk api_server --a2a --port 8001 contributing/samples/a2a_human_in_loop/remote_a2a
   ```

2. **Run the Main Agent**:
   ```bash
   # In a separate terminal, run the adk web server
   adk web contributing/samples/
   ```

### Example Interactions

Once both services are running, you can interact with the root agent through the approval workflow:

**Automatic Approval (Under $100):**
```
User: Please reimburse $50 for meals
Agent: I'll process your reimbursement request for $50 for meals. Since this amount is under $100, I can approve it automatically.
Agent: ✅ Reimbursement approved and processed: $50 for meals
```

**Human Approval Required (Over $100):**
```
User: Please reimburse $200 for conference travel
Agent: I'll process your reimbursement request for $200 for conference travel. Since this amount exceeds $100, I need to get manager approval.
Agent: 🔄 Request submitted for approval (Ticket: reimbursement-ticket-001). Please wait for manager review.
[Human manager interacts with root agent to approve the request]
Agent: ✅ Great news! Your reimbursement has been approved by the manager. Processing $200 for conference travel.
```

## Code Structure

### Main Agent (`agent.py`)

- **`reimburse(purpose: str, amount: float)`**: Function tool for processing reimbursements
- **`approval_agent`**: Remote A2A agent configuration for human approval workflows
- **`root_agent`**: Main reimbursement agent with automatic/manual approval logic

### Remote Approval Agent (`remote_a2a/human_in_loop/`)

- **`agent.py`**: Implementation of the approval agent with long-running tools
- **`agent.json`**: Agent card of the A2A agent

- **`ask_for_approval()`**: Long-running tool that handles approval requests

## Long-Running Tool Workflow

The human-in-the-loop process follows this pattern:

1. **Initial Call**: Root agent delegates approval request to remote approval agent for amounts >$100
2. **Pending Response**: Remote approval agent returns immediate response with `status: "pending"` and ticket ID and serface the approval request to root agent
3. **Agent Acknowledgment**: Root agent informs user about pending approval status
4. **Human Interaction**: Human manager interacts with root agent to review and approve/reject the request
5. **Updated Response**: Root agent receives updated tool response with approval decision and send it to remote agent
6. **Final Action**: Remote agent processes the approval and completes the reimbursement and send the result to root_agent

## Extending the Sample

You can extend this sample by:

- Adding more complex approval hierarchies (multiple approval levels)
- Implementing different approval rules based on expense categories
- Creating additional remote agent for budget checking or policy validation
- Adding notification systems for approval status updates
- Integrating with external approval systems or databases
- Implementing approval timeouts and escalation procedures

## Deployment to Other Environments

When deploying the remote approval A2A agent to different environments (e.g., Cloud Run, different hosts/ports), you **must** update the `url` field in the agent card JSON file:

### Local Development
```json
{
  "url": "http://localhost:8001/a2a/human_in_loop",
  ...
}
```

### Cloud Run Example
```json
{
  "url": "https://your-approval-service-abc123-uc.a.run.app/a2a/human_in_loop",
  ...
}
```

### Custom Host/Port Example
```json
{
  "url": "https://your-domain.com:9000/a2a/human_in_loop",
  ...
}
```

**Important:** The `url` field in `remote_a2a/human_in_loop/agent.json` must point to the actual RPC endpoint where your remote approval A2A agent is deployed and accessible.

## Troubleshooting

**Connection Issues:**
- Ensure the local ADK web server is running on port 8000
- Ensure the remote A2A server is running on port 8001
- Check that no firewall is blocking localhost connections
- **Verify the `url` field in `remote_a2a/human_in_loop/agent.json` matches the actual deployed location of your remote A2A server**
- Verify the agent card URL passed to RemoteA2AAgent constructor matches the running A2A server

**Agent Not Responding:**
- Check the logs for both the local ADK web server on port 8000 and remote A2A server on port 8001
- Verify the agent instructions are clear and unambiguous
- Ensure long-running tool responses are properly formatted with matching IDs
- **Double-check that the RPC URL in the agent.json file is correct and accessible**

**Approval Workflow Issues:**
- Verify that updated tool responses use the same `id` and `name` as the original function call
- Check that the approval status is correctly updated in the tool response
- Ensure the human approval process is properly simulated or integrated
