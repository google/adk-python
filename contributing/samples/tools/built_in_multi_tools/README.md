This agent is to demonstrate that the built-in google search tool and the
VertexAiSearchTool can be used together with other tools, even though the model
has the limitation that built-in tool cannot be used by other tools.

It is achieved by the workarounds added in https://github.com/google/adk-python/blob/4485379a049a5c84583a43c85d444ea1f1ba6f12/src/google/adk/agents/llm_agent.py#L124-L149.

To run this agent, set the environment variable `VERTEXAI_DATASTORE_ID`
(e.g.
`projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}`)
and use `adk web`.

You can follow
https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es
to set up the datastore.
