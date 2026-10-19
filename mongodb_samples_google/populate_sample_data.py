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

"""Populates the sample database with realistic agent sessions.

Google-embedding twin of mongodb_samples/populate_sample_data.py: creates
multi-turn conversations for fictional users of the mongodb_samples_google
product-search agent. Each turn executes the real search tools against Atlas
in the default embedding mode — the tools embed each query with the Google
embedding model (Vertex AI `text-embedding-005`) and send the vector to
`$vectorSearch`. Every session is then ingested into long-term memory.

Fills all five integration collections: sessions, events, app_states,
user_states, and memories. Documents belonging to the sample app
(`app_name = "mongodb_samples_google"`) are wiped first, so re-running the
script replaces the demo data instead of piling up duplicates.

Prerequisites:
  1. MONGODB_URI / MONGODB_DATABASE are set (see .env); the Google-embedding
     collection defaults to `products_google` (MONGODB_GOOGLE_COLLECTION).
  2. GOOGLE_CLOUD_PROJECT is set so the tools can embed queries.
  3. The collection has been seeded with embeddings and its search indexes
     are READY: uv run python setup_mongodb_sample.py

Usage (from the repository root):
  set -a && source .env && set +a
  uv run python mongodb_samples_google/populate_sample_data.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time

# The repository root holds setup_mongodb_sample.py (the embedding helper).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from google.adk.events.event import Event  # noqa: E402
from google.adk.events.event_actions import EventActions  # noqa: E402
from google.adk.integrations.mongodb import _search_tool  # noqa: E402
from google.adk.integrations.mongodb import MongoDbMemoryService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbSessionService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbToolSettings  # noqa: E402
from google.genai import types  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from setup_mongodb_sample import _make_genai_client  # noqa: E402

APP = "mongodb_samples_google"
AGENT = "mongodb_google_search_agent"

# The Google-embedding collection holds an `embedding` field produced by
# Vertex AI `text-embedding-005`; queries are embedded with the same model.
_SEARCH_SETTINGS = MongoDbToolSettings(
    vertex_ai_embedding_model_name="text-embedding-005"
)

# (user_id, session topic state, [(user message, search kind, tool query)])
CONVERSATIONS = [
    (
        "alice",
        {"topic": "vacuums", "app:catalog_version": "2026.10"},
        [(
            "I'm looking for a cordless robot vacuum for pet hair",
            "vector",
            "cordless robot vacuum for pet hair",
        )],
        {
            "user:preferred_category": "cleaning",
            "user:last_query": "robot vacuum",
        },
    ),
    (
        "alice",
        {"topic": "coffee"},
        [(
            "Which espresso machine comes with a milk frother?",
            "hybrid",
            "espresso coffee machine with milk frother",
        )],
        {"user:last_query": "espresso machine"},
    ),
    (
        "bob",
        {"topic": "kitchen"},
        [(
            "What do you have for baking bread?",
            "vector",
            "stand mixer for baking bread",
        )],
        {
            "user:preferred_category": "kitchen",
            "user:last_query": "stand mixer",
        },
    ),
    (
        "carol",
        {"topic": "outdoor"},
        [(
            "I need a backpack for weekend hikes",
            "hybrid",
            "lightweight hiking backpack with rain cover",
        )],
        {
            "user:preferred_category": "outdoor",
            "user:last_query": "hiking backpack",
        },
    ),
]


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    sys.exit(f"Environment variable {name} is required. See .env.")
  return value


def _user_event(text: str, invocation: str, timestamp: float) -> Event:
  return Event(
      invocation_id=invocation,
      author="user",
      content=types.Content(role="user", parts=[types.Part(text=text)]),
      timestamp=timestamp,
  )


def _tool_call_event(
    tool_name: str, args: dict, invocation: str, timestamp: float
) -> Event:
  return Event(
      invocation_id=invocation,
      author=AGENT,
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=f"call-{invocation}", name=tool_name, args=args
                  )
              )
          ],
      ),
      timestamp=timestamp,
  )


def _tool_response_event(
    tool_name: str, result: dict, invocation: str, timestamp: float
) -> Event:
  return Event(
      invocation_id=invocation,
      author=AGENT,
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=f"call-{invocation}", name=tool_name, response=result
                  )
              )
          ],
      ),
      timestamp=timestamp,
  )


def _agent_event(
    text: str, invocation: str, timestamp: float, state_delta: dict
) -> Event:
  return Event(
      invocation_id=invocation,
      author=AGENT,
      content=types.Content(role="model", parts=[types.Part(text=text)]),
      actions=EventActions(state_delta=state_delta),
      timestamp=timestamp,
  )


async def _run_search(
    kind: str,
    query: str,
    client: MongoClient,
    genai_client,
    database: str,
    collection: str,
) -> tuple[str, dict, dict]:
  """Executes the real search tool and returns (tool name, args, result)."""
  if kind == "vector":
    args = {"collection_name": collection, "query": query, "limit": 3}
    result = await _search_tool.vector_search(
        **args,
        client=client,
        database_name=database,
        settings=_SEARCH_SETTINGS,
        genai_client=genai_client,
    )
    return "mongodb_vector_search", args, result
  args = {
      "collection_name": collection,
      "query": query,
      "text_search_field": "description",
      "limit": 3,
  }
  result = await _search_tool.hybrid_search(
      **args,
      client=client,
      database_name=database,
      settings=_SEARCH_SETTINGS,
      genai_client=genai_client,
  )
  return "mongodb_hybrid_search", args, result


def _summarize(result: dict) -> str:
  rows = result.get("rows", [])
  if not rows:
    return "I searched the catalog but found no matching products."
  lines = [
      f"- {row.get('name')} (score {row.get('search_score'):.3f})"
      for row in rows
  ]
  return "Here's what I found in the catalog:\n" + "\n".join(lines)


async def main() -> None:
  uri = _require_env("MONGODB_URI")
  database = _require_env("MONGODB_DATABASE")
  collection = os.environ.get("MONGODB_GOOGLE_COLLECTION", "products_google")

  genai_client = _make_genai_client()
  if genai_client is None:
    sys.exit(
        "Set GOOGLE_CLOUD_PROJECT (or a gcloud project) so the tools can"
        " embed queries with the Google embedding model."
    )

  client = MongoClient(uri)
  client.admin.command("ping")
  print(f"Connected. Database: {database}, collection: {collection}")

  # Reset only the sample app's documents so re-runs stay idempotent.
  db = client[database]
  db["events"].delete_many({"app_name": APP})
  db["sessions"].delete_many({"_id": {"$regex": f"^{APP}/"}})
  db["app_states"].delete_many({"_id": APP})
  db["user_states"].delete_many({"_id": {"$regex": f"^{APP}/"}})
  db["memories"].delete_many({"app_name": APP})

  session_service = MongoDbSessionService(
      connection_string=uri, database_name=database
  )
  memory_service = MongoDbMemoryService(
      connection_string=uri, database_name=database
  )

  clock = time.time()
  try:
    for user, session_state, turns, turn_delta in CONVERSATIONS:
      session = await session_service.create_session(
          app_name=APP, user_id=user, state=session_state
      )
      for user_text, kind, tool_query in turns:
        clock += 4
        invocation = f"inv-{user}-{int(clock)}"
        await session_service.append_event(
            session, _user_event(user_text, invocation, clock)
        )

        tool_name, args, result = await _run_search(
            kind, tool_query, client, genai_client, database, collection
        )
        await session_service.append_event(
            session, _tool_call_event(tool_name, args, invocation, clock + 1)
        )
        await session_service.append_event(
            session,
            _tool_response_event(tool_name, result, invocation, clock + 2),
        )
        await session_service.append_event(
            session,
            _agent_event(_summarize(result), invocation, clock + 3, turn_delta),
        )
      print(
          f"session {session.id}: {user} /"
          f" {session_state['topic']} -> {len(session.events)} events"
      )

      reloaded = await session_service.get_session(
          app_name=APP, user_id=user, session_id=session.id
      )
      await memory_service.add_session_to_memory(reloaded)
  finally:
    await session_service.close()
    await memory_service.close()

  print("\nCollection document counts:")
  for name in (
      "sessions",
      "events",
      "app_states",
      "user_states",
      "memories",
      collection,
  ):
    print(f"  {name}: {db[name].count_documents({})}")

  client.close()


if __name__ == "__main__":
  asyncio.run(main())
