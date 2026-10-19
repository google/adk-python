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

"""Seeds the sample MongoDB database used by the mongodb_samples agent.

Creates two collections holding the same ten sample products, one per
embedding mode supported by the MongoDB search tools:

  1. `MONGODB_COLLECTION` (default "products") — Atlas Automated Embedding
     (Preview). Documents carry plain text only; an `autoEmbed` vector index
     over `description` makes Atlas generate embeddings with a Voyage AI
     model at index time and query time. Query it with
     `MongoDbToolSettings(use_mongodb_auto_embedding=True)`.
  2. `MONGODB_GOOGLE_COLLECTION` (default "products_google") — Google
     embedding models. Documents carry an `embedding` field produced here by
     Vertex AI `text-embedding-005`, under a regular `vector` index. Query
     it with default settings; the tools embed each query with the same
     Google model through the genai client.

Both collections also get a dynamic full-text search index (`default`) so
`mongodb_hybrid_search` works against either one.

Configuration is read from environment variables:
  MONGODB_URI:              MongoDB connection string (required), e.g.
                            "mongodb+srv://user:pass@cluster.mongodb.net/".
  MONGODB_DATABASE:         Database to create (default:
                            "mongodb_sample_db").
  MONGODB_COLLECTION:       Auto-embedding collection (default: "products").
  MONGODB_GOOGLE_COLLECTION: Google-embedding collection (default:
                            "products_google").
  GOOGLE_CLOUD_PROJECT:     GCP project for Vertex AI embeddings. Falls back
                            to `gcloud config get-value project`. If no
                            project/credentials are available, the Google
                            collection is skipped with a warning.
  GOOGLE_CLOUD_LOCATION:    Vertex AI region (default: "us-central1").

Usage (from the repository root):
  set -a && source .env && set +a
  uv run python setup_mongodb_sample.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

_VECTOR_INDEX_NAME = "vector_index"  # MongoDbToolSettings default
_SEARCH_INDEX_NAME = "default"  # MongoDbToolSettings default
_INDEX_READY_TIMEOUT_SECONDS = 300

_GOOGLE_EMBEDDING_MODEL = "text-embedding-005"
_GOOGLE_EMBEDDING_DIMENSIONS = 768  # native size of text-embedding-005

_PRODUCTS = [
    {
        "name": "Robot Vacuum Pro",
        "category": "cleaning",
        "price": 299.99,
        "description": (
            "Cordless robot vacuum with laser navigation, strong suction for"
            " pet hair, and automatic dirt disposal."
        ),
    },
    {
        "name": "Steam Mop Elite",
        "category": "cleaning",
        "price": 89.5,
        "description": (
            "Lightweight steam mop that sanitizes hardwood and tile floors"
            " without chemicals."
        ),
    },
    {
        "name": "Air Fryer Max",
        "category": "kitchen",
        "price": 129.0,
        "description": (
            "Digital air fryer with eight cooking presets and a"
            " dishwasher-safe non-stick basket."
        ),
    },
    {
        "name": "Espresso Maker Barista",
        "category": "kitchen",
        "price": 349.0,
        "description": (
            "Compact espresso machine with 15-bar pump, milk frother, and"
            " stainless steel finish."
        ),
    },
    {
        "name": "Noise-Cancel Headphones",
        "category": "electronics",
        "price": 199.99,
        "description": (
            "Over-ear wireless headphones with active noise cancellation and"
            " 40-hour battery life."
        ),
    },
    {
        "name": "Smart Speaker Mini",
        "category": "electronics",
        "price": 49.0,
        "description": (
            "Voice-controlled smart speaker with room-filling sound and"
            " built-in assistant."
        ),
    },
    {
        "name": "Dog Chew Toy Set",
        "category": "pets",
        "price": 24.95,
        "description": (
            "Durable rubber chew toys for aggressive chewers, pack of five"
            " in assorted shapes."
        ),
    },
    {
        "name": "Cat Scratching Tower",
        "category": "pets",
        "price": 79.99,
        "description": (
            "Multi-level cat tree with sisal scratching posts, hammock, and"
            " dangling toys."
        ),
    },
    {
        "name": "Camping Tent 4P",
        "category": "outdoor",
        "price": 159.0,
        "description": (
            "Waterproof four-person camping tent with quick-setup poles and"
            " vestibule."
        ),
    },
    {
        "name": "Hiking Backpack 40L",
        "category": "outdoor",
        "price": 119.95,
        "description": (
            "Lightweight 40-liter hiking backpack with rain cover and"
            " hydration sleeve."
        ),
    },
]


def _wait_for_indexes(collection, expected_names: set[str]) -> None:
  """Polls until all expected search indexes report READY."""
  deadline = time.monotonic() + _INDEX_READY_TIMEOUT_SECONDS
  while time.monotonic() < deadline:
    statuses = {
        index["name"]: index.get("status")
        for index in collection.list_search_indexes()
        if index["name"] in expected_names
    }
    if statuses.keys() == expected_names and all(
        status == "READY" for status in statuses.values()
    ):
      return
    time.sleep(5)
  raise TimeoutError(
      f"Search indexes {expected_names} not READY within"
      f" {_INDEX_READY_TIMEOUT_SECONDS}s: {statuses}"
  )


def _ensure_full_text_index(collection) -> None:
  """Creates the dynamic full-text index used by hybrid search."""
  existing = {index["name"] for index in collection.list_search_indexes()}
  if _SEARCH_INDEX_NAME in existing:
    return
  collection.create_search_index(
      model=SearchIndexModel(
          definition={"mappings": {"dynamic": True}},
          name=_SEARCH_INDEX_NAME,
          type="search",
      )
  )
  print(f"  Created full-text search index '{_SEARCH_INDEX_NAME}'.")


def _index_has_auto_embed(index: dict) -> bool:
  """True when the search index definition contains an autoEmbed field."""
  definition = index.get("latestDefinition", {})
  return any(
      field.get("type") == "autoEmbed" for field in definition.get("fields", [])
  )


def _seed_auto_embed_collection(database, collection_name: str) -> None:
  """Seeds the Atlas Automated Embedding collection (no vectors stored).

  Documents hold plain text only. The `autoEmbed` index makes Atlas embed
  `description` with a Voyage AI model at index time and embed each query at
  query time.
  """
  collection = database[collection_name]
  collection.delete_many({})
  collection.insert_many(_PRODUCTS)
  print(f"  Inserted {len(_PRODUCTS)} documents (text only).")

  for index in collection.list_search_indexes():
    # A leftover `vector`-type index from an earlier seed (embeddings stored
    # in the documents) cannot serve text queries; replace it.
    if index["name"] == _VECTOR_INDEX_NAME and not _index_has_auto_embed(index):
      collection.drop_search_index(_VECTOR_INDEX_NAME)
      print(f"  Dropped legacy vector index '{_VECTOR_INDEX_NAME}'.")
  existing = {index["name"] for index in collection.list_search_indexes()}
  if _VECTOR_INDEX_NAME not in existing:
    collection.create_search_index(
        model=SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "autoEmbed",
                        "modality": "text",
                        "path": "description",
                        "model": "voyage-4",
                    },
                    {"type": "filter", "path": "category"},
                ]
            },
            name=_VECTOR_INDEX_NAME,
            type="vectorSearch",
        )
    )
    print(f"  Created auto-embedding index '{_VECTOR_INDEX_NAME}' (voyage-4).")
  _ensure_full_text_index(collection)
  _wait_for_indexes(collection, {_VECTOR_INDEX_NAME, _SEARCH_INDEX_NAME})


def _make_genai_client():
  """Builds a Vertex AI genai client from the environment or gcloud ADC.

  Returns None when no project can be determined, so the Google-embedding
  collection is skipped rather than failing the whole seed.
  """
  project = os.environ.get("GOOGLE_CLOUD_PROJECT")
  if not project:
    try:
      project = subprocess.run(
          ["gcloud", "config", "get-value", "project"],
          capture_output=True,
          text=True,
          check=False,
      ).stdout.strip()
    except OSError:
      project = ""
  if not project:
    return None
  from google.genai import Client  # pylint: disable=import-outside-toplevel

  return Client(
      vertexai=True,
      project=project,
      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
  )


def _seed_google_embedding_collection(database, collection_name: str) -> None:
  """Seeds the Google embedding collection with text-embedding-005 vectors."""
  genai_client = _make_genai_client()
  if genai_client is None:
    print(
        f"  Skipping {collection_name}: no GOOGLE_CLOUD_PROJECT or gcloud"
        " project found. Set one and re-run to seed the Google embedding"
        " collection."
    )
    return

  texts = [p["name"] + " " + p["description"] for p in _PRODUCTS]
  try:
    response = genai_client.models.embed_content(
        model=_GOOGLE_EMBEDDING_MODEL, contents=texts
    )
  except Exception as exc:  # API disabled, quota, auth — keep the rest usable.
    print(f"  Skipping {collection_name}: embedding failed ({exc!r}).")
    return
  embeddings = [e.values for e in response.embeddings]

  collection = database[collection_name]
  collection.delete_many({})
  collection.insert_many([
      {**product, "embedding": list(embedding)}
      for product, embedding in zip(_PRODUCTS, embeddings)
  ])
  print(
      f"  Inserted {len(_PRODUCTS)} documents with {_GOOGLE_EMBEDDING_MODEL}"
      " embeddings."
  )

  existing = {index["name"] for index in collection.list_search_indexes()}
  if _VECTOR_INDEX_NAME not in existing:
    collection.create_search_index(
        model=SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": _GOOGLE_EMBEDDING_DIMENSIONS,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "category"},
                ]
            },
            name=_VECTOR_INDEX_NAME,
            type="vectorSearch",
        )
    )
    print(f"  Created vector search index '{_VECTOR_INDEX_NAME}'.")
  _ensure_full_text_index(collection)
  _wait_for_indexes(collection, {_VECTOR_INDEX_NAME, _SEARCH_INDEX_NAME})


def main() -> None:
  uri = os.environ.get("MONGODB_URI")
  if not uri:
    sys.exit("Set the MONGODB_URI environment variable before running.")
  database_name = os.environ.get("MONGODB_DATABASE", "mongodb_sample_db")
  auto_collection = os.environ.get("MONGODB_COLLECTION", "products")
  google_collection = os.environ.get(
      "MONGODB_GOOGLE_COLLECTION", "products_google"
  )

  client = MongoClient(uri)
  client.admin.command("ping")
  print(f"Connected to cluster. Seeding {database_name} ...")
  database = client[database_name]

  print(f"\n[{auto_collection}] Atlas Automated Embedding (Voyage AI):")
  _seed_auto_embed_collection(database, auto_collection)

  print(f"\n[{google_collection}] Google embedding models (Vertex AI):")
  _seed_google_embedding_collection(database, google_collection)

  print("\nDone. Sample database is ready for mongodb_samples.")
  client.close()


if __name__ == "__main__":
  main()
