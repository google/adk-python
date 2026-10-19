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

"""Generates reviewable sample data by driving the deployed Agent Engine apps.

Sends a battery of realistic product questions to both deployed engines
(auto-embedding and Google-embedding), mirrors every interaction into MongoDB
as sessions/events/state (app "mongodb_samples" / "mongodb_samples_google"),
and ingests them into long-term memory in BOTH modes:

  - `memories`        — keyword-mode recall (default).
  - `memories_vector` — vector-mode recall (embeddings + vector index).

Nothing is dropped or deleted: every document stays in the sample database
for review in the Atlas UI.

Usage (from the repository root):
  set -a && source .env && set +a
  export GOOGLE_CLOUD_PROJECT=gcp-pov
  uv run python mongodb_samples/generate_review_data.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time

# The repository root holds setup_mongodb_sample.py (the embedding helper).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from google.adk.events.event import Event  # noqa: E402
from google.adk.events.event_actions import EventActions  # noqa: E402
from google.adk.integrations.mongodb import MongoDbMemoryService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbSessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pymongo import MongoClient  # noqa: E402
import vertexai  # noqa: E402
from vertexai import agent_engines  # noqa: E402

from setup_mongodb_sample import _make_genai_client  # noqa: E402

_ENGINES = {
    "mongodb_samples": "projects/787220387490/locations/us-central1/reasoningEngines/3509280750924988416",
    "mongodb_samples_google": "projects/787220387490/locations/us-central1/reasoningEngines/404048807853031424",
}

# (query, expected product substring in the answer or tool rows)
_QUERIES = [
    ("I need a cordless vacuum that handles pet hair", "Robot Vacuum Pro"),
    ("Which espresso machine has a milk frother?", "Espresso Maker Barista"),
    ("What do you have for camping in the rain?", "Camping Tent 4P"),
    ("Looking for headphones with long battery life", "Noise-Cancel Headphones"),
    ("What's a good gift for a cat owner?", "Cat Scratching Tower"),
    ("Something lightweight for weekend hiking", "Hiking Backpack 40L"),
    ("Do you have kitchen appliances for frying without oil?", "Air Fryer Max"),
    ("What can clean hardwood floors without chemicals?", "Steam Mop Elite"),
    ("I need a smart speaker for my kitchen", "Smart Speaker Mini"),
    ("Toys for a dog that chews through everything", "Dog Chew Toy Set"),
]


def _events_of(engine_response_events: list[dict]) -> dict:
  """Splits streamed AE events into (user text, tool calls, tool responses, final text)."""
  tool_calls = []
  tool_responses = []
  final_text = ""
  for event in engine_response_events:
    content = event.get("content", {})
    for part in content.get("parts", []):
      if "function_call" in part:
        tool_calls.append(part["function_call"])
      elif "function_response" in part:
        tool_responses.append(part["function_response"])
      elif part.get("text"):
        final_text += part["text"]
  return {
      "tool_calls": tool_calls,
      "tool_responses": tool_responses,
      "final_text": final_text,
  }


async def _run_query(engine, user_id: str, query: str) -> tuple[dict, float]:
  """Runs one query in its own session; returns (parsed events, latency)."""
  session = await engine.async_create_session(user_id=user_id)
  sid = session["id"] if isinstance(session, dict) else session.id
  events = []
  start = time.perf_counter()
  async for event in engine.async_stream_query(
      user_id=user_id, session_id=sid, message=query
  ):
    events.append(event)
  return _events_of(events), time.perf_counter() - start


def _to_mongodb_events(
    query: str, parsed: dict, invocation: str, clock: float, agent_name: str
) -> list[Event]:
  """Mirrors one AE interaction as ADK events in the populate script's shape.

  Event ids are deterministic (`<invocation>-<n>`), so re-runs upsert the
  same memory documents instead of piling up duplicates.
  """
  events = [
      Event(
          id=f"{invocation}-0",
          invocation_id=invocation,
          author="user",
          content=types.Content(role="user", parts=[types.Part(text=query)]),
          timestamp=clock,
      )
  ]
  for i, call in enumerate(parsed["tool_calls"]):
    events.append(
        Event(
            id=f"{invocation}-tc{i}",
            invocation_id=invocation,
            author=agent_name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call.get("id", f"call-{invocation}-{i}"),
                            name=call["name"],
                            args=call.get("args", {}),
                        )
                    )
                ],
            ),
            timestamp=clock + 1 + i * 0.1,
        )
    )
  for i, response in enumerate(parsed["tool_responses"]):
    events.append(
        Event(
            id=f"{invocation}-tr{i}",
            invocation_id=invocation,
            author=agent_name,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=response.get("id", f"call-{invocation}-{i}"),
                            name=response["name"],
                            response=response.get("response", {}),
                        )
                    )
                ],
            ),
            timestamp=clock + 1.5 + i * 0.1,
        )
    )
  events.append(
      Event(
          id=f"{invocation}-final",
          invocation_id=invocation,
          author=agent_name,
          content=types.Content(
              role="model", parts=[types.Part(text=parsed["final_text"])]
          ),
          actions=EventActions(
              state_delta={"user:last_query": query[:80]}
          ),
          timestamp=clock + 2,
      )
  )
  return events


async def main() -> None:
  uri = os.environ["MONGODB_URI"]
  database = os.environ["MONGODB_DATABASE"]
  vertexai.init(
      project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1"
  )

  client = MongoClient(uri)
  session_service = MongoDbSessionService(
      connection_string=uri, database_name=database
  )
  keyword_memory = MongoDbMemoryService(
      connection_string=uri, database_name=database
  )
  vector_memory = MongoDbMemoryService(
      connection_string=uri,
      database_name=database,
      memories_collection="memories_vector",
      retrieval_mode="vector",
      genai_client=_make_genai_client(),
  )

  # Clean up leftovers from interrupted runs of this script (they used
  # random session ids and would duplicate the deterministic re-run).
  db = client[database]
  db["sessions"].delete_many({"user_id": {"$regex": "^review-user-"}})
  db["events"].delete_many(
      {"app_name": {"$in": list(_ENGINES)}, "user_id": {"$regex": "^review-user-"}}
  )
  db["user_states"].delete_many({"_id": {"$regex": "/review-user-"}})
  for coll in ("memories", "memories_vector"):
    db[coll].delete_many({"user_id": {"$regex": "^review-user-"}})

  report = []
  clock = time.time()
  query_sem = asyncio.Semaphore(4)  # bound concurrent engine calls

  async def handle_query(
      engine, app_name: str, agent_name: str, i: int, query: str, expected: str
  ) -> dict:
    async with query_sem:
      parsed, latency = await _run_query(engine, f"review-user-{i}", query)
    clock_i = clock + i * 5
    invocation = f"ae-{app_name}-{i}"

    # Persist the interaction as a MongoDB session (never deleted; the
    # deterministic session/event ids make re-runs upsert, not duplicate).
    session_id = f"review-{app_name}-{i}"
    try:
      session = await session_service.create_session(
          app_name=app_name,
          user_id=f"review-user-{i}",
          session_id=session_id,
          state={"topic": "review", "app:source": "agent_engine"},
      )
    except Exception:
      session = await session_service.get_session(
          app_name=app_name, user_id=f"review-user-{i}", session_id=session_id
      )
    for event in _to_mongodb_events(
        query, parsed, invocation, clock_i, agent_name
    ):
      await session_service.append_event(session, event)
    reloaded = await session_service.get_session(
        app_name=app_name, user_id=f"review-user-{i}", session_id=session_id
    )
    # Ingest into both memory modes (idempotent upserts).
    await keyword_memory.add_session_to_memory(reloaded)
    await vector_memory.add_session_to_memory(reloaded)

    rows = []
    for response in parsed["tool_responses"]:
      rows.extend(response.get("response", {}).get("rows", []))
    hit_names = [row.get("name", "") for row in rows]
    tool_names = [c["name"] for c in parsed["tool_calls"]]
    ok = any(expected in name for name in hit_names) or expected in parsed[
        "final_text"
    ]
    print(
        f"  [{'OK' if ok else 'MISS'}] {query[:52]:<52} tools={tool_names}"
        f" top={hit_names[:1]} {latency:.1f}s"
    )
    return {
        "app": app_name,
        "query": query,
        "expected": expected,
        "tools": tool_names,
        "top_hits": hit_names[:3],
        "ok": ok,
        "latency_s": round(latency, 1),
    }

  for app_name, resource in _ENGINES.items():
    agent_name = (
        "mongodb_search_agent"
        if app_name == "mongodb_samples"
        else "mongodb_google_search_agent"
    )
    engine = agent_engines.get(resource)
    print(f"\n=== {app_name} ({resource.split('/')[-1]}) ===")
    report.extend(
        await asyncio.gather(*(
            handle_query(engine, app_name, agent_name, i, query, expected)
            for i, (query, expected) in enumerate(_QUERIES)
        ))
    )

  # Nothing is deleted — report the final collection state for review.
  db = client[database]
  print("\n=== collections left in Atlas for review ===")
  counts = {}
  for name in (
      "sessions",
      "events",
      "app_states",
      "user_states",
      "memories",
      "memories_vector",
      "products",
      "products_google",
  ):
    counts[name] = db[name].count_documents({})
    print(f"  {name}: {counts[name]}")

  out = Path(__file__).resolve().parent / "review_data_report.json"
  out.write_text(
      json.dumps({"queries": report, "collection_counts": counts}, indent=2),
      encoding="utf-8",
  )
  print(f"\nReport written to {out}")

  await session_service.close()
  await keyword_memory.close()
  await vector_memory.close()
  client.close()


if __name__ == "__main__":
  asyncio.run(main())
