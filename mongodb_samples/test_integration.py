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

"""Live integration test for the MongoDB integration (no LLM required).

Exercises every component of `google.adk.integrations.mongodb` against a real
MongoDB deployment:

  - MongoDbToolset: tool exposure via get_tools().
  - mongodb_vector_search: live $vectorSearch query, including a filtered
    query, against the seeded products collection.
  - mongodb_hybrid_search: live $rankFusion query combining full-text and
    vector search.
  - MongoDbSessionService: create/get/list sessions, append events with
    app:/user:/session-scoped state deltas, and delete.
  - MongoDbMemoryService: ingest a session into memory and recall it by
    keyword search.

The script cleans up every document it creates (test sessions, events, state
and memory documents use a dedicated app name), so it is safe to re-run.

Prerequisites:
  1. MONGODB_URI / MONGODB_DATABASE / MONGODB_COLLECTION are set (see .env).
  2. The products collection has been seeded and its search indexes are READY:
       uv run python setup_mongodb_sample.py

Usage (from the repository root):
  set -a && source .env && set +a
  uv run python mongodb_samples/test_integration.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

# The repository root holds setup_mongodb_sample.py (the embedding helper).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from google.adk.events.event import Event  # noqa: E402
from google.adk.events.event_actions import EventActions  # noqa: E402
from google.adk.integrations.mongodb import _search_tool  # noqa: E402
from google.adk.integrations.mongodb import MongoDbMemoryService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbSessionService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbToolset  # noqa: E402
from google.adk.integrations.mongodb import MongoDbToolSettings  # noqa: E402
from google.adk.sessions.base_session_service import GetSessionConfig  # noqa: E402
from google.genai import types  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from setup_mongodb_sample import _make_genai_client  # noqa: E402

# The `products` collection uses Atlas Automated Embedding: queries go out as
# plain text against the `description` autoEmbed field, and Atlas embeds them
# with the index's Voyage AI model.
_AUTO_SETTINGS = MongoDbToolSettings(
    use_mongodb_auto_embedding=True, default_embedding_field="description"
)

# Dedicated app/user so the test never touches real application data and can
# clean up after itself reliably.
APP = "mongodb_samples_itest"
USER = "itest_user"

_FAILURES: list[str] = []


def _check(label: str, condition: bool, detail: str = "") -> None:
  """Records a pass/fail line for the final report."""
  status = "PASS" if condition else "FAIL"
  suffix = f" ({detail})" if detail and not condition else ""
  print(f"  [{status}] {label}{suffix}")
  if not condition:
    _FAILURES.append(label)


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    sys.exit(
        f"Environment variable {name} is required. See"
        " mongodb_samples/README.md for setup instructions."
    )
  return value


def _text_event(text: str, *, author: str = "user", timestamp: float) -> Event:
  return Event(
      invocation_id="itest",
      author=author,
      content=types.Content(
          role="user" if author == "user" else "model",
          parts=[types.Part(text=text)],
      ),
      timestamp=timestamp,
  )


async def test_toolset(client: MongoClient, database: str) -> None:
  """MongoDbToolset exposes the two search tools with the mongodb prefix."""
  print("\n== MongoDbToolset ==")
  toolset = MongoDbToolset(
      connection_string=_require_env("MONGODB_URI"), database_name=database
  )
  try:
    tools = await toolset.get_tools()
    names = sorted(tool.name for tool in tools)
    _check(
        "get_tools returns vector_search and hybrid_search",
        names == ["hybrid_search", "vector_search"],
        f"got {names}",
    )
    prefixed = sorted(
        tool.name for tool in await toolset.get_tools_with_prefix()
    )
    _check(
        "tools are exposed to the agent as mongodb_vector_search and"
        " mongodb_hybrid_search",
        prefixed == ["mongodb_hybrid_search", "mongodb_vector_search"],
        f"got {prefixed}",
    )
  finally:
    await toolset.close()

  # Constructor argument validation.
  try:
    MongoDbToolset(database_name=database)
    _check("toolset requires a client or connection string", False)
  except ValueError:
    _check("toolset requires a client or connection string", True)
  try:
    MongoDbToolset(
        database_name=database,
        connection_string=_require_env("MONGODB_URI"),
        mongo_client=client,
    )
    _check("toolset rejects client + connection string together", False)
  except ValueError:
    _check("toolset rejects client + connection string together", True)


async def test_vector_search(
    client: MongoClient, database: str, collection: str
) -> None:
  """Live $vectorSearch against `products` in Atlas auto-embedding mode."""
  print("\n== mongodb_vector_search (Atlas auto-embedding) ==")

  result = await _search_tool.vector_search(
      collection_name=collection,
      query="cordless robot vacuum for pet hair",
      client=client,
      database_name=database,
      settings=_AUTO_SETTINGS,
      limit=3,
  )
  _check(
      "vector search returns SUCCESS",
      result["status"] == "SUCCESS",
      str(result.get("error_details")),
  )
  rows = result.get("rows", [])
  _check(
      "vector search respects limit=3", len(rows) == 3, f"got {len(rows)} rows"
  )
  top_name = rows[0].get("name") if rows else None
  _check(
      "vacuum query ranks Robot Vacuum Pro first",
      top_name == "Robot Vacuum Pro",
      f"top hit: {top_name}",
  )
  _check(
      "every row carries a search_score",
      all("search_score" in row for row in rows),
  )
  _check(
      "auto-embedding mode keeps the searched text field in results",
      all("description" in row for row in rows),
  )

  filtered = await _search_tool.vector_search(
      collection_name=collection,
      query="cordless robot vacuum for pet hair",
      client=client,
      database_name=database,
      settings=_AUTO_SETTINGS,
      filter={"category": "kitchen"},
      limit=5,
  )
  filtered_rows = filtered.get("rows", [])
  _check(
      "filtered vector search returns only kitchen products",
      filtered["status"] == "SUCCESS"
      and bool(filtered_rows)
      and all(row.get("category") == "kitchen" for row in filtered_rows),
      f"rows: {[row.get('name') for row in filtered_rows]}",
  )


async def test_google_embedding_mode(
    client: MongoClient, database: str, collection: str
) -> None:
  """Live $vectorSearch against `products_google` with text-embedding-005.

  The default embedding mode: the tool embeds the query with the Google
  model through the genai client and sends the vector as `queryVector`.
  Skipped (not failed) when the collection was not seeded, e.g. because no
  Google credentials were available to setup_mongodb_sample.py.
  """
  print("\n== mongodb_vector_search (Google embedding models) ==")
  if client[database][collection].count_documents({}) == 0:
    print(
        f"  [SKIP] {database}.{collection} is empty; run"
        " setup_mongodb_sample.py with Google credentials to seed it."
    )
    return

  genai_client = _make_genai_client()
  if genai_client is None:
    print("  [SKIP] no Google credentials available for query embedding.")
    return

  result = await _search_tool.vector_search(
      collection_name=collection,
      query="cordless robot vacuum for pet hair",
      genai_client=genai_client,
      client=client,
      database_name=database,
      settings=MongoDbToolSettings(),
      limit=3,
  )
  _check(
      "google-mode vector search returns SUCCESS",
      result["status"] == "SUCCESS",
      str(result.get("error_details")),
  )
  rows = result.get("rows", [])
  top_name = rows[0].get("name") if rows else None
  _check(
      "vacuum query ranks Robot Vacuum Pro first",
      top_name == "Robot Vacuum Pro",
      f"top hit: {top_name}",
  )
  _check(
      "embedding vector is excluded from results by default",
      rows and all("embedding" not in row for row in rows),
  )
  _check(
      "text fields stay in results",
      rows and all("description" in row for row in rows),
  )


async def test_hybrid_search(
    client: MongoClient, database: str, collection: str
) -> None:
  """Live $rankFusion query in auto-embedding mode (text + vector arms)."""
  print("\n== mongodb_hybrid_search (Atlas auto-embedding) ==")

  result = await _search_tool.hybrid_search(
      collection_name=collection,
      query="espresso coffee machine with milk frother",
      text_search_field="description",
      client=client,
      database_name=database,
      settings=_AUTO_SETTINGS,
      limit=3,
  )
  _check(
      "hybrid search returns SUCCESS",
      result["status"] == "SUCCESS",
      str(result.get("error_details")),
  )
  rows = result.get("rows", [])
  _check("hybrid search returns rows", bool(rows))
  names = [row.get("name") for row in rows]
  _check(
      "espresso query surfaces Espresso Maker Barista",
      "Espresso Maker Barista" in names,
      f"rows: {names}",
  )
  _check(
      "every row carries a fused search_score",
      all("search_score" in row for row in rows),
  )


async def test_session_service(uri: str, database: str) -> None:
  """End-to-end MongoDbSessionService flow with state scoping and cleanup."""
  print("\n== MongoDbSessionService ==")
  service = MongoDbSessionService(connection_string=uri, database_name=database)
  try:
    session = await service.create_session(
        app_name=APP,
        user_id=USER,
        state={
            "topic": "vacuums",
            "app:deployment": "itest",
            "user:preferred_category": "cleaning",
        },
    )
    _check(
        "create_session returns merged scoped state",
        session.state.get("topic") == "vacuums"
        and session.state.get("app:deployment") == "itest"
        and session.state.get("user:preferred_category") == "cleaning",
        f"state: {session.state}",
    )

    event = _text_event(
        "I am looking for a robot vacuum for pet hair", timestamp=1000.0
    )
    event.actions = EventActions(
        state_delta={"turn": 1, "user:last_query": "robot vacuum"}
    )
    await service.append_event(session, event)

    fetched = await service.get_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    _check(
        "get_session round-trips the appended event",
        fetched is not None
        and len(fetched.events) == 1
        and fetched.events[0].content.parts[0].text
        == "I am looking for a robot vacuum for pet hair",
    )
    _check(
        "state delta merged with scopes on reload",
        fetched is not None
        and fetched.state.get("turn") == 1
        and fetched.state.get("user:last_query") == "robot vacuum"
        and fetched.state.get("app:deployment") == "itest",
        f"state: {fetched.state if fetched else None}",
    )

    listed = await service.list_sessions(app_name=APP, user_id=USER)
    _check(
        "list_sessions includes the test session",
        any(s.id == session.id for s in listed.sessions),
    )

    user_state = await service.get_user_state(app_name=APP, user_id=USER)
    _check(
        "get_user_state returns user: scoped entries",
        user_state.get("preferred_category") == "cleaning"
        and user_state.get("last_query") == "robot vacuum",
        f"user state: {user_state}",
    )

    empty = await service.get_session(
        app_name=APP,
        user_id=USER,
        session_id=session.id,
        config=GetSessionConfig(num_recent_events=0),
    )
    _check(
        "num_recent_events=0 probes existence without events",
        empty is not None and empty.events == [],
    )

    await service.delete_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    gone = await service.get_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    _check("delete_session removes the session", gone is None)
  finally:
    await service.close()


async def test_memory_service(
    uri: str, database: str, cleanup_client: MongoClient
) -> None:
  """End-to-end MongoDbMemoryService flow: ingest a session, recall it."""
  print("\n== MongoDbMemoryService ==")
  session_service = MongoDbSessionService(
      connection_string=uri, database_name=database
  )
  memory_service = MongoDbMemoryService(
      connection_string=uri, database_name=database
  )
  try:
    session = await session_service.create_session(app_name=APP, user_id=USER)
    await session_service.append_event(
        session,
        _text_event(
            "My favorite appliance is the espresso machine with the milk"
            " frother.",
            timestamp=2000.0,
        ),
    )
    reloaded = await session_service.get_session(
        app_name=APP, user_id=USER, session_id=session.id
    )

    await memory_service.add_session_to_memory(reloaded)

    response = await memory_service.search_memory(
        app_name=APP, user_id=USER, query="espresso machine"
    )
    texts = [
        " ".join(part.text for part in m.content.parts if part.text)
        for m in response.memories
    ]
    _check(
        "search_memory recalls the ingested event by keyword",
        any("espresso machine" in text for text in texts),
        f"memories: {texts}",
    )

    other_user = await memory_service.search_memory(
        app_name=APP, user_id="someone_else", query="espresso machine"
    )
    _check(
        "memories are scoped per user",
        not other_user.memories,
    )

    # Idempotency: re-ingesting the same session must not duplicate memories.
    await memory_service.add_session_to_memory(reloaded)
    again = await memory_service.search_memory(
        app_name=APP, user_id=USER, query="espresso machine"
    )
    _check(
        "re-ingesting a session does not duplicate memories",
        len(again.memories) == len(response.memories),
        f"{len(response.memories)} -> {len(again.memories)}",
    )
  finally:
    # Cleanup: remove the test session and every memory document for the
    # dedicated test app, then close the service-owned clients.
    await session_service.delete_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    await session_service.close()
    await memory_service.close()
    cleanup_client[database]["memories"].delete_many({"app_name": APP})
    cleanup_client[database]["app_states"].delete_many({"_id": APP})
    cleanup_client[database]["user_states"].delete_many(
        {"_id": f"{APP}/{USER}"}
    )


async def main() -> int:
  uri = _require_env("MONGODB_URI")
  database = _require_env("MONGODB_DATABASE")
  collection = _require_env("MONGODB_COLLECTION")
  google_collection = os.environ.get(
      "MONGODB_GOOGLE_COLLECTION", "products_google"
  )

  client = MongoClient(uri)
  client.admin.command("ping")
  print(f"Connected to MongoDB. Database: {database}, collection: {collection}")

  index_names = {
      index["name"]
      for index in client[database][collection].list_search_indexes()
  }
  settings = MongoDbToolSettings()
  expected = {
      settings.default_vector_index_name,
      settings.default_search_index_name,
  }
  if not expected.issubset(index_names):
    sys.exit(
        f"Search indexes {expected} not found on {database}.{collection} "
        f"(found: {sorted(index_names)}). Run `uv run python"
        " setup_mongodb_sample.py` first."
    )

  await test_toolset(client, database)
  await test_vector_search(client, database, collection)
  await test_hybrid_search(client, database, collection)
  await test_google_embedding_mode(client, database, google_collection)
  await test_session_service(uri, database)
  await test_memory_service(uri, database, client)

  client.close()

  print("\n" + "=" * 60)
  if _FAILURES:
    print(f"RESULT: {len(_FAILURES)} check(s) FAILED:")
    for failure in _FAILURES:
      print(f"  - {failure}")
    return 1
  print("RESULT: all checks passed.")
  return 0


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
