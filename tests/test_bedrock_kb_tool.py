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

from unittest.mock import MagicMock, patch

from google.adk.tools.bedrock_kb_tool import BedrockKBTool
from google.genai import types as genai_types
import pytest


class TestBedrockKBToolInit:
    """Tests for BedrockKBTool initialization."""

    def test_init_with_explicit_params(self):
        tool = BedrockKBTool(
            knowledge_base_id="KB123",
            region_name="us-west-2",
            number_of_results=10,
        )

        assert tool.knowledge_base_id == "KB123"
        assert tool.region_name == "us-west-2"
        assert tool.number_of_results == 10
        assert tool.name == "bedrock_knowledge_base"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_defaults(self):
        """With no params and no env vars, region defaults to us-east-1."""
        tool = BedrockKBTool(knowledge_base_id="KB456")

        assert tool.knowledge_base_id == "KB456"
        assert tool.region_name == "us-east-1"
        assert tool.number_of_results == 5

    @patch.dict("os.environ", {"KNOWLEDGE_BASE_ID": "ENV_KB_ID", "AWS_REGION": "eu-west-1"})
    def test_init_from_env_vars(self):
        tool = BedrockKBTool()

        assert tool.knowledge_base_id == "ENV_KB_ID"
        assert tool.region_name == "eu-west-1"

    @patch.dict("os.environ", {"KNOWLEDGE_BASE_ID": "ENV_KB_ID", "AWS_REGION": "eu-west-1"})
    def test_explicit_params_override_env_vars(self):
        tool = BedrockKBTool(knowledge_base_id="EXPLICIT_ID", region_name="ap-southeast-1")

        assert tool.knowledge_base_id == "EXPLICIT_ID"
        assert tool.region_name == "ap-southeast-1"


class TestBedrockKBToolDeclaration:
    """Tests for _get_declaration."""

    def test_get_declaration_returns_correct_schema(self):
        tool = BedrockKBTool(knowledge_base_id="KB123")
        declaration = tool._get_declaration()

        assert isinstance(declaration, genai_types.FunctionDeclaration)
        assert declaration.name == "bedrock_knowledge_base"
        assert "knowledge base" in declaration.description.lower()
        assert declaration.parameters.type == "OBJECT"
        assert "query" in declaration.parameters.properties
        assert declaration.parameters.properties["query"].type == "STRING"
        assert declaration.parameters.required == ["query"]


class TestBedrockKBToolRunAsync:
    """Tests for run_async method."""

    @pytest.fixture
    def mock_boto3_client(self):
        """Create a mock client and return it for direct injection into tool._client."""
        mock_client = MagicMock()
        return mock_client

    def _make_tool(self, mock_client, **kwargs):
        """Create a BedrockKBTool with agentic retrieval disabled and inject mock client."""
        kwargs.setdefault("use_agentic_retrieval", False)
        tool = BedrockKBTool(**kwargs)
        tool._client = mock_client
        return tool

    @pytest.mark.asyncio
    async def test_run_async_managed_config(self, mock_boto3_client):
        tool = self._make_tool(
            mock_boto3_client,
            knowledge_base_id="KB_MANAGED",
            number_of_results=3,
        )

        mock_boto3_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Document content 1"},
                    "location": {"s3Location": {"uri": "s3://bucket/doc1.pdf"}},
                    "score": 0.95,
                },
                {
                    "content": {"text": "Document content 2"},
                    "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                    "score": 0.85,
                },
            ]
        }

        result = await tool.run_async(args={"query": "test query"})

        mock_boto3_client.retrieve.assert_called_once_with(
            knowledgeBaseId="KB_MANAGED",
            retrievalQuery={"text": "test query"},
            retrievalConfiguration={
                "managedSearchConfiguration": {"numberOfResults": 3}
            },
        )

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["content"] == "Document content 1"
        assert result["results"][0]["source"] == "s3://bucket/doc1.pdf"
        assert result["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_run_async_empty_results(self, mock_boto3_client):
        tool = self._make_tool(mock_boto3_client, knowledge_base_id="KB_EMPTY")

        mock_boto3_client.retrieve.return_value = {"retrievalResults": []}

        result = await tool.run_async(args={"query": "no matches"})

        assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_run_async_empty_query_returns_error(self, mock_boto3_client):
        tool = self._make_tool(mock_boto3_client, knowledge_base_id="KB123")

        result = await tool.run_async(args={"query": ""})

        assert "error" in result
        assert "No query provided" in result["error"]
        mock_boto3_client.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_missing_query_returns_error(self, mock_boto3_client):
        tool = self._make_tool(mock_boto3_client, knowledge_base_id="KB123")

        result = await tool.run_async(args={})

        assert "error" in result
        assert "No query provided" in result["error"]
        mock_boto3_client.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_api_error(self, mock_boto3_client):
        tool = self._make_tool(mock_boto3_client, knowledge_base_id="KB_ERROR")

        mock_boto3_client.retrieve.side_effect = Exception(
            "AccessDeniedException: Not authorized"
        )

        result = await tool.run_async(args={"query": "will fail"})

        assert "error" in result
        assert "Error retrieving from Bedrock KB" in result["error"]
        assert "AccessDeniedException" in result["error"]

    @pytest.mark.asyncio
    async def test_run_async_partial_result_fields(self, mock_boto3_client):
        """Results with missing optional fields should use defaults."""
        tool = self._make_tool(mock_boto3_client, knowledge_base_id="KB_PARTIAL")

        mock_boto3_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Partial doc"},
                    # No location or score
                },
            ]
        }

        result = await tool.run_async(args={"query": "partial"})

        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Partial doc"
        assert result["results"][0]["source"] == ""
        assert result["results"][0]["score"] == 0.0
