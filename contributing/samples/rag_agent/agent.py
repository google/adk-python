# Copyright 2025 Google LLC
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

import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

load_dotenv()

ask_vertex_retrieval = VertexAiRagRetrieval(
    name="retrieve_rag_documentation",
    description=(
        "Use this tool to retrieve documentation and reference materials for"
        " the question from the RAG corpus,"
    ),
    rag_resources=[
        rag.RagResource(
            # please fill in your own rag corpus
            # e.g. projects/123/locations/us-central1/ragCorpora/456
            rag_corpus=os.environ.get("RAG_CORPUS"),
        )
    ],
    similarity_top_k=1,
    vector_distance_threshold=0.6,
)

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="root_agent",
    instruction=(
        "You are an AI assistant with access to specialized corpus of"
        " documents. Your role is to provide accurate and concise answers to"
        " questions based on documents that are retrievable using"
        " ask_vertex_retrieval."
    ),
    tools=[ask_vertex_retrieval],
)
