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

"""Sample agent using FilesRetrieval with gemini-embedding-2-preview.

This agent indexes local text files and answers questions about them
using retrieval-augmented generation.

Usage:
  cd contributing/samples
  adk run files_retrieval_agent
  # or
  adk web .
"""

import os

from google.adk.agents.llm_agent import Agent
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

files_retrieval = FilesRetrieval(
    name="search_documents",
    description=(
        "Search through local ADK documentation files to find relevant"
        " information. Use this tool when the user asks questions about ADK"
        " features, architecture, or tools."
    ),
    input_dir=DATA_DIR,
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="files_retrieval_agent",
    instruction=(
        "You are a helpful assistant that answers questions about the Agent"
        " Development Kit (ADK). Use the search_documents tool to find"
        " relevant information before answering. Always base your answers"
        " on the retrieved documents."
    ),
    tools=[files_retrieval],
)
