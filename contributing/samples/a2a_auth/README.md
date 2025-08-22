# A2A OAuth Authentication Sample Agent

This sample demonstrates the **Agent-to-Agent (A2A)** architecture with **OAuth Authentication** workflows in the Agent Development Kit (ADK). The sample implements a multi-agent system where a remote agent can surface OAuth authentication requests to the local agent, which then guides the end user through the OAuth flow before returning the authentication credentials to the remote agent for API access.

## Overview

The A2A OAuth Authentication sample consists of:

- **Root Agent** (`root_agent`): The main orchestrator that handles user requests and delegates tasks to specialized agents
- **YouTube Search Agent** (`youtube_search_agent`): A local agent that handles YouTube video searches using LangChain tools
- **BigQuery Agent** (`bigquery_agent`): A remote A2A agent that manages BigQuery operations and requires OAuth authentication for Google Cloud access

## Architecture

```
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│   End User      │───▶│   Root Agent       │───▶│   BigQuery Agent │
│   (OAuth Flow)  │    │    (Local)         │    │  (Remote A2A)    │
│                 │    │                    │    │ (localhost:8001) │
│   OAuth UI      │◀───│                    │◀───│   OAuth Request  │
└─────────────────┘    └────────────────────┘    └──────────────────┘
```

## Key Features

### 1. **Multi-Agent Architecture**
- Root agent coordinates between local YouTube search and remote BigQuery operations
- Demonstrates hybrid local/remote agent workflows
- Seamless task delegation based on user request types

### 2. **OAuth Authentication Workflow**
- Remote BigQuery agent surfaces OAuth authentication requests to the root agent
- Root agent guides end users through Google OAuth flow for BigQuery access
- Secure token exchange between agents for authenticated API calls

### 3. **Google Cloud Integration**
- BigQuery toolset with comprehensive dataset and table management capabilities
- OAuth-protected access to user's Google Cloud BigQuery resources
- Support for listing, creating, and managing datasets and tables

### 4. **LangChain Tool Integration**
- YouTube search functionality using LangChain community tools
- Demonstrates integration of third-party tools in agent workflows

## Setup and Usage

### Prerequisites

1. **Set up OAuth Credentials**:
   ```bash
   export OAUTH_CLIENT_ID=your_google_oauth_client_id
   export OAUTH_CLIENT_SECRET=your_google_oauth_client_secret
   ```

2. **Start the Remote BigQuery Agent server**:
   ```bash
   # Start the remote a2a server that serves the BigQuery agent on port 8001
   adk api_server --a2a --port 8001 contributing/samples/a2a_auth/remote_a2a
   ```

3. **Run the Main Agent**:
   ```bash
   # In a separate terminal, run the adk web server
   adk web contributing/samples/
   ```

### Example Interactions

Once both services are running, you can interact with the root agent:

**YouTube Search (No Authentication Required):**
```
User: Search for 3 Taylor Swift music videos
Agent: I'll help you search for Taylor Swift music videos on YouTube.
[Agent delegates to YouTube Search Agent]
Agent: I found 3 Taylor Swift music videos:
1. "Anti-Hero" - Official Music Video
2. "Shake It Off" - Official Music Video
3. "Blank Space" - Official Music Video
```

**BigQuery Operations (OAuth Required):**
```
User: List my BigQuery datasets
Agent: I'll help you access your BigQuery datasets. This requires authentication with your Google account.
[Agent delegates to BigQuery Agent]
Agent: To access your BigQuery data, please complete the OAuth authentication.
[OAuth flow initiated - user redirected to Google authentication]
User: [Completes OAuth flow in browser]
Agent: Authentication successful! Here are your BigQuery datasets:
- dataset_1: Customer Analytics
- dataset_2: Sales Data
- dataset_3: Marketing Metrics
```

**Dataset Management:**
```
User: Show me details for my Customer Analytics dataset
Agent: I'll get the details for your Customer Analytics dataset.
[Using existing OAuth token]
Agent: Customer Analytics Dataset Details:
- Created: 2024-01-15
- Location: US
- Tables: 5
- Description: Customer behavior and analytics data
```

## Code Structure

### Main Agent (`agent.py`)

- **`youtube_search_agent`**: Local agent with LangChain YouTube search tool
- **`bigquery_agent`**: Remote A2A agent configuration for BigQuery operations
- **`root_agent`**: Main orchestrator with task delegation logic

### Remote BigQuery Agent (`remote_a2a/bigquery_agent/`)

- **`agent.py`**: Implementation of the BigQuery agent with OAuth toolset
- **`agent.json`**: Agent card of the A2A agent
- **`BigQueryToolset`**: OAuth-enabled tools for BigQuery dataset and table management

## OAuth Authentication Workflow

The OAuth authentication process follows this pattern:

1. **Initial Request**: User requests BigQuery operation through root agent
2. **Delegation**: Root agent delegates to remote BigQuery agent
3. **Auth Check**: BigQuery agent checks for valid OAuth token
4. **Auth Request**: If no token, agent surfaces OAuth request to root agent
5. **User OAuth**: Root agent guides user through Google OAuth flow
6. **Token Exchange**: Root agent sends OAuth token to BigQuery agent
7. **API Call**: BigQuery agent uses token to make authenticated API calls
8. **Result Return**: BigQuery agent returns results through root agent to user

## Supported BigQuery Operations

The BigQuery agent supports the following operations:

### Dataset Operations:
- **List Datasets**: `bigquery_datasets_list` - Get all user's datasets
- **Get Dataset**: `bigquery_datasets_get` - Get specific dataset details
- **Create Dataset**: `bigquery_datasets_insert` - Create new dataset

### Table Operations:
- **List Tables**: `bigquery_tables_list` - Get tables in a dataset
- **Get Table**: `bigquery_tables_get` - Get specific table details
- **Create Table**: `bigquery_tables_insert` - Create new table in dataset

## Extending the Sample

You can extend this sample by:

- Adding more Google Cloud services (Cloud Storage, Compute Engine, etc.)
- Implementing token refresh and expiration handling
- Adding role-based access control for different BigQuery operations
- Creating OAuth flows for other providers (Microsoft, Facebook, etc.)
- Adding audit logging for authentication events
- Implementing multi-tenant OAuth token management

## Deployment to Other Environments

When deploying the remote BigQuery A2A agent to different environments (e.g., Cloud Run, different hosts/ports), you **must** update the `url` field in the agent card JSON file:

### Local Development
```json
{
  "url": "http://localhost:8001/a2a/bigquery_agent",
  ...
}
```

### Cloud Run Example
```json
{
  "url": "https://your-bigquery-service-abc123-uc.a.run.app/a2a/bigquery_agent",
  ...
}
```

### Custom Host/Port Example
```json
{
  "url": "https://your-domain.com:9000/a2a/bigquery_agent",
  ...
}
```

**Important:** The `url` field in `remote_a2a/bigquery_agent/agent.json` must point to the actual RPC endpoint where your remote BigQuery A2A agent is deployed and accessible.

## Troubleshooting

**Connection Issues:**
- Ensure the local ADK web server is running on port 8000
- Ensure the remote A2A server is running on port 8001
- Check that no firewall is blocking localhost connections
- **Verify the `url` field in `remote_a2a/bigquery_agent/agent.json` matches the actual deployed location of your remote A2A server**
- Verify the agent card URL passed to RemoteA2AAgent constructor matches the running A2A server


**OAuth Issues:**
- Verify OAuth client ID and secret are correctly set in .env file
- Ensure OAuth redirect URIs are properly configured in Google Cloud Console
- Check that the OAuth scopes include BigQuery access permissions
- Verify the user has access to the BigQuery projects/datasets

**BigQuery Access Issues:**
- Ensure the authenticated user has BigQuery permissions
- Check that the Google Cloud project has BigQuery API enabled
- Verify dataset and table names are correct and accessible
- Check for quota limits on BigQuery API calls

**Agent Communication Issues:**
- Check the logs for both the local ADK web server and remote A2A server
- Verify OAuth tokens are properly passed between agents
- Ensure agent instructions are clear about authentication requirements
- **Double-check that the RPC URL in the agent.json file is correct and accessible**
