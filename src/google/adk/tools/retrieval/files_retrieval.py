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

"""Provides data for the agent."""

from __future__ import annotations

import logging

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

from .llama_index_retrieval import LlamaIndexRetrieval

logger = logging.getLogger("google_adk." + __name__)


class FilesRetrieval(LlamaIndexRetrieval):

  def __init__(self, *, name: str, description: str, input_dir: str):

    self.input_dir = input_dir

    logger.info("Loading data from %s", input_dir)
    retriever = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(input_dir).load_data()
    ).as_retriever()
    super().__init__(name=name, description=description, retriever=retriever)
