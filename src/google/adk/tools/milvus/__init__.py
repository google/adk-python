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

"""Milvus Tools (Experimental).

Milvus Tools provide vector similarity search capabilities using Milvus
as the vector database backend. This module offers:

1. A MilvusToolset for easy integration with ADK agents.
2. A MilvusVectorStore utility for collection management and data ingestion.
3. Support for any third-party embedding function (Google GenAI, OpenAI, etc.).
"""

from .milvus_toolset import MilvusToolset

__all__ = [
    "MilvusToolset",
]
