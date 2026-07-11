# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Amazon Bedrock Knowledge Base retrieval tool for Google ADK.

Provides a tool that queries Amazon Bedrock Managed Knowledge Bases
for use in ADK agents.

Usage:
    from google.adk.tools.bedrock_kb_tool import BedrockKBTool

    kb_tool = BedrockKBTool(knowledge_base_id="ABCDEFGHIJ")
    agent = Agent(tools=[kb_tool])
"""

import os
from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get('location', {})
    loc_type = location.get('type', '')
    if loc_type == 'S3' or 's3Location' in location:
        return location.get('s3Location', {}).get('uri', '')
    elif loc_type == 'WEB' or 'webLocation' in location:
        return location.get('webLocation', {}).get('url', '')
    elif 'confluenceLocation' in location:
        return location.get('confluenceLocation', {}).get('url', '')
    elif 'salesforceLocation' in location:
        return location.get('salesforceLocation', {}).get('url', '')
    elif 'sharePointLocation' in location:
        return location.get('sharePointLocation', {}).get('url', '')
    elif 'customDocumentLocation' in location:
        return location.get('customDocumentLocation', {}).get('id', '')
    # Fallback to metadata._source_uri (for agentic results)
    return result.get('metadata', {}).get('_source_uri', '')


class BedrockKBTool(BaseTool):
    """Retrieves documents from an Amazon Bedrock Managed Knowledge Base.

    Args:
        knowledge_base_id: The KB ID. Falls back to KNOWLEDGE_BASE_ID env var.
        region_name: AWS region. Falls back to AWS_REGION env var or us-east-1.
        number_of_results: Max results to return. Defaults to 5.
        use_agentic_retrieval: Use AgenticRetrieveStream for complex queries with
            query decomposition and managed reranking. Falls back to plain Retrieve
            on failure. Defaults to True.
    """

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        region_name: Optional[str] = None,
        number_of_results: int = 5,
        use_agentic_retrieval: Optional[bool] = None,
    ):
        super().__init__(
            name="bedrock_knowledge_base",
            description=(
                "Retrieves relevant documents from an Amazon Bedrock Knowledge Base. "
                "Use this to search for information in the knowledge base."
            ),
        )
        self.knowledge_base_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID", "")
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.number_of_results = number_of_results
        self.use_agentic_retrieval = use_agentic_retrieval if use_agentic_retrieval is not None else os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false'
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockKBTool. "
                    "Install with: pip install boto3>=1.43.2"
                )
            self._client = boto3.client(
                "bedrock-agent-runtime",
                region_name=self.region_name,
                config=Config(user_agent_extra="google-adk/bedrock-kb"),
            )
        return self._client

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        """Return the function declaration for this tool."""
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "query": genai_types.Schema(
                        type="STRING",
                        description="The search query to find relevant documents.",
                    )
                },
                required=["query"],
            ),
        )

    def _managed_retrieve(self, query: str) -> dict[str, Any]:
        """Retrieve using plain managed Retrieve API."""
        client = self._get_client()

        retrieval_config = {
            "managedSearchConfiguration": {
                "numberOfResults": self.number_of_results
            }
        }

        response = client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration=retrieval_config,
        )

        results = []
        for result in response.get("retrievalResults", []):
            content = result.get("content", {}).get("text", "")
            source = _get_source_uri(result)
            score = result.get("score", 0.0)
            results.append({
                "content": content,
                "source": source,
                "score": score,
            })

        return {"results": results}

    def _agentic_retrieve(self, query: str) -> dict[str, Any]:
        """Retrieve using AgenticRetrieveStream with fallback to plain Retrieve."""
        try:
            client = self._get_client()
            response = client.agentic_retrieve_stream(
                knowledgeBaseId=self.knowledge_base_id,
                messages=[{"content": {"text": query}, "role": "user"}],
                retrievers=[{
                    "configuration": {
                        "knowledgeBase": {
                            "knowledgeBaseId": self.knowledge_base_id,
                            "retrievalOverrides": {
                                "maxNumberOfResults": self.number_of_results
                            },
                        }
                    }
                }],
                agenticRetrieveConfiguration={
                    "foundationModelType": "MANAGED",
                    "rerankingModelType": "MANAGED",
                },
            )
            # Process streaming response
            results = []
            for event in response.get("stream", []):
                if "result" in event and "results" in event["result"]:
                    for result in event["result"]["results"]:
                        content = result.get("content", {}).get("text", "")
                        source = _get_source_uri(result)
                        score = result.get("score", 0.0)
                        results.append({
                            "content": content,
                            "source": source,
                            "score": score,
                        })
            return {"results": results}
        except Exception:
            # Fall back to plain managed retrieve
            return self._managed_retrieve(query)

    async def run_async(self, *, args: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute the retrieval."""
        query = args.get("query", "")
        if not query:
            return {"error": "No query provided."}

        try:
            if self.use_agentic_retrieval:
                return self._agentic_retrieve(query)
            return self._managed_retrieve(query)
        except Exception as e:
            return {"error": f"Error retrieving from Bedrock KB: {e}"}
