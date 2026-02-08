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

"""Sample Milvus RAG Agent.

This sample demonstrates how to build a knowledge base agent using Milvus
as the vector database for retrieval-augmented generation (RAG).

Prerequisites:
  1. A running Milvus instance (or use Milvus Lite with a local file path).
  2. A Google GenAI API key set in the GOOGLE_API_KEY environment variable.
  3. Install dependencies: pip install "google-adk[milvus]"

Usage:
  adk run contributing/samples/milvus_rag_agent
"""

import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.milvus.milvus_toolset import MilvusToolset
from google.adk.tools.milvus.settings import MilvusToolSettings
from google.adk.tools.milvus.settings import MilvusVectorStoreSettings
from google.genai import Client

load_dotenv()

# --- Embedding function using Google GenAI ---
genai_client = Client()


def embedding_fn(texts: list[str]) -> list[list[float]]:
  response = genai_client.models.embed_content(
      model="text-embedding-004",
      contents=texts,
  )
  return [list(e.values) for e in response.embeddings]


# --- Milvus vector store settings ---
# Replace these with your own Milvus connection and collection settings.
vector_store_settings = MilvusVectorStoreSettings(
    # Use a remote Milvus instance or a local Milvus Lite file path.
    uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
    token=os.environ.get("MILVUS_TOKEN", None),
    collection_name=os.environ.get("MILVUS_COLLECTION", "knowledge_base"),
    dimension=768,
    metric_type="COSINE",
    top_k=5,
)

# --- Milvus toolset ---
milvus_toolset = MilvusToolset(
    milvus_tool_settings=MilvusToolSettings(
        vector_store_settings=vector_store_settings,
    ),
    embedding_fn=embedding_fn,
    tool_filter=["similarity_search"],
)

# --- Agent definition ---
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="milvus_knowledge_agent",
    description="Agent that answers questions using a Milvus knowledge base.",
    instruction="""
    You are a helpful assistant with access to a knowledge base.
    1. Always use the `similarity_search` tool to find relevant information.
    2. Present the search results naturally in your response.
    3. If no results are found, say you don't know.
    """,
    tools=[milvus_toolset],
)
