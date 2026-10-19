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

"""Sample agent that uses the MongoDB integration with Google embedding models.

This is the Google-embedding twin of `mongodb_samples` (which tests Atlas
Automated Embedding). Here every search embeds the query text with the Vertex
AI `text-embedding-005` model through the genai client and sends the vector to
`$vectorSearch` as `queryVector`, against documents whose `embedding` field
was produced by the same model.

Configuration is read from environment variables:
  MONGODB_URI:        MongoDB connection string, e.g.
                      "mongodb+srv://user:pass@cluster.mongodb.net/".
  MONGODB_DATABASE:   The database the search tools operate on.
  MONGODB_GOOGLE_COLLECTION: The collection the agent should search (default
                      "products_google").
  GOOGLE_CLOUD_PROJECT: GCP project for Vertex AI query embeddings.
  GOOGLE_CLOUD_LOCATION: Vertex AI region (default "us-central1").

Requires the pymongo package: pip install google-adk[mongodb].
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.integrations.mongodb import MongoDbToolset
from google.adk.integrations.mongodb import MongoDbToolSettings
from google.genai import Client


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f"Environment variable {name} is required to run this sample."
        " See mongodb_samples_google/README.md for setup instructions."
    )
  return value


MONGODB_URI = _require_env("MONGODB_URI")
MONGODB_DATABASE = _require_env("MONGODB_DATABASE")
MONGODB_COLLECTION = os.environ.get(
    "MONGODB_GOOGLE_COLLECTION", "products_google"
)

# The sample collection was seeded by setup_mongodb_sample.py with embeddings
# from Vertex AI `text-embedding-005`. The toolset embeds each query with the
# same model through this client, so the query vector is comparable with the
# stored vectors. `default_embedding_field` keeps its default ("embedding"),
# the document field holding the vectors.
genai_client = Client(
    vertexai=True,
    project=_require_env("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# `connection_string_env_var` keeps the URI off the pickled toolset: only the
# variable's name is stored, so Agent Engine deployment packages carry no
# credential (deploy_agent_engine.py injects the value from Secret Manager).
mongodb_toolset = MongoDbToolset(
    connection_string_env_var="MONGODB_URI",
    database_name=MONGODB_DATABASE,
    genai_client=genai_client,
    settings=MongoDbToolSettings(
        vertex_ai_embedding_model_name="text-embedding-005"
    ),
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="mongodb_google_search_agent",
    description=(
        "Answers questions by searching documents in a MongoDB collection,"
        " embedding queries with a Google embedding model."
    ),
    instruction=f"""You are a search assistant backed by MongoDB.

You have exactly two tools: `mongodb_vector_search` and
`mongodb_hybrid_search`. Never call any other tool name.

Both tools take the user's question as plain text in `query` and embed it
for you with the Google embedding model. When the user asks about products or
documents:

1. Call `mongodb_vector_search` for semantic "similar to ..." queries, or
   `mongodb_hybrid_search` (also pass `text_search_field="description"`)
   when both keyword and semantic matching help.
2. Always use collection_name `{MONGODB_COLLECTION}`.

Summarize the returned rows for the user and mention each row's
`search_score`.
""",
    tools=[mongodb_toolset],
)
