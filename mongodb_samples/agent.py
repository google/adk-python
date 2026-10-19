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

"""Sample agent that uses the MongoDB integration (vector + hybrid search).

Configuration is read from environment variables:
  MONGODB_URI:        MongoDB connection string, e.g.
                      "mongodb+srv://user:pass@cluster.mongodb.net/".
  MONGODB_DATABASE:   The database the search tools operate on.
  MONGODB_COLLECTION: The collection the agent should search by default.

Requires the pymongo package: pip install google-adk[mongodb].
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.integrations.mongodb import MongoDbToolset
from google.adk.integrations.mongodb import MongoDbToolSettings


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f"Environment variable {name} is required to run this sample."
        " See mongodb_samples/README.md for setup instructions."
    )
  return value


MONGODB_URI = _require_env("MONGODB_URI")  # local runs (adk run / adk web)
MONGODB_DATABASE = _require_env("MONGODB_DATABASE")
MONGODB_COLLECTION = _require_env("MONGODB_COLLECTION")

# The sample collection uses Atlas Automated Embedding (Preview): an
# `autoEmbed` index over the `description` text field makes Atlas generate
# embeddings with a Voyage AI model, so the tools send each query as plain
# text and no Google embedding credentials are needed.
# `default_embedding_field` names the autoEmbed text field the tools search.
#
# `connection_string_env_var` keeps the URI off the pickled toolset: only the
# variable's name is stored, so Agent Engine deployment packages carry no
# credential (deploy_agent_engine.py injects the value from Secret Manager).
mongodb_toolset = MongoDbToolset(
    connection_string_env_var="MONGODB_URI",
    database_name=MONGODB_DATABASE,
    settings=MongoDbToolSettings(
        use_mongodb_auto_embedding=True,
        default_embedding_field="description",
    ),
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="mongodb_search_agent",
    description=(
        "Answers questions by searching documents in a MongoDB collection."
    ),
    instruction=f"""You are a search assistant backed by MongoDB.

You have exactly two tools: `mongodb_vector_search` and
`mongodb_hybrid_search`. Never call any other tool name.

Both tools take the user's question as plain text in `query` and embed it
for you. When the user asks about products or documents:

1. Call `mongodb_vector_search` for semantic "similar to ..." queries, or
   `mongodb_hybrid_search` (also pass `text_search_field="description"`)
   when both keyword and semantic matching help.
2. Always use collection_name `{MONGODB_COLLECTION}`.

Summarize the returned rows for the user and mention each row's
`search_score`.
""",
    tools=[mongodb_toolset],
)
