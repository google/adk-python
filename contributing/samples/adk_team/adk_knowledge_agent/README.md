# Agent Knowledge Agent

An intelligent assistant for performing Vertex AI Search to find ADK knowledge
and documentation.

## Deployment

This agent is deployed to Google Could Run as an A2A agent, which is used by
the parent ADK Agent Builder Assistant.

Here are the steps to deploy the agent:

1. Set environment variables

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1 # Or your preferred location
export GOOGLE_GENAI_USE_ENTERPRISE=True
export VERTEXAI_DATASTORE_ID=your-datastore-id
```

`VERTEXAI_DATASTORE_ID` is required, and is the full resource name of the
Vertex AI Search data store the agent queries, with the format
`projects/{project_number}/locations/{location}/collections/{collection}/dataStores/{datastore_id}`.
The agent refuses to start without it.

2. Run the deployment command

```bash
$ adk deploy cloud_run --project=your-project-id --region=us-central1 --service_name=adk-agent-builder-knowledge-service --with_ui --a2a ./adk_knowledge_agent
```
