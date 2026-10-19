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

"""Load test for the MongoDB integration, run against a live deployment.

Measures latency and throughput of every component of
`google.adk.integrations.mongodb` under increasing client concurrency:

  1. mongodb_vector_search in Atlas auto-embedding mode (`query.text`).
  2. mongodb_hybrid_search in Atlas auto-embedding mode ($rankFusion).
  3. mongodb_vector_search in Google embedding mode (Vertex AI embed per op).
  4. MongoDbSessionService round trip: create + append_event + get + delete.
  5. MongoDbMemoryService: add_session_to_memory + search_memory.

All state-writing scenarios use a dedicated app name and clean up every
document they create, so the script is safe to re-run.

Prerequisites:
  1. MONGODB_URI / MONGODB_DATABASE / MONGODB_COLLECTION are set (see .env).
  2. The collections are seeded and search indexes are READY:
       uv run python setup_mongodb_sample.py
  3. For scenario 3, Google credentials usable for Vertex AI embeddings
     (GOOGLE_CLOUD_PROJECT or a gcloud default project).

Usage (from the repository root):
  set -a && source .env && set +a
  uv run python mongodb_samples/load_test.py

Writes load_test_report.md and load_test_results.json next to this script.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import getpass
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

# The repository root holds setup_mongodb_sample.py (the embedding helper).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from google.adk.events.event import Event  # noqa: E402
from google.adk.integrations.mongodb import _search_tool  # noqa: E402
from google.adk.integrations.mongodb import MongoDbMemoryService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbSessionService  # noqa: E402
from google.adk.integrations.mongodb import MongoDbToolSettings  # noqa: E402
from google.genai import types  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from setup_mongodb_sample import _make_genai_client  # noqa: E402

_APP = "mongodb_samples_loadtest"

# Atlas auto-embedding settings for the `products` collection; default
# (Google embedding) settings for the `products_google` collection.
_AUTO_SETTINGS = MongoDbToolSettings(
    use_mongodb_auto_embedding=True, default_embedding_field="description"
)
_GOOGLE_SETTINGS = MongoDbToolSettings(
    vertex_ai_embedding_model_name="text-embedding-005"
)

_CONCURRENCY_LEVELS = [1, 5, 10, 20]
_SEARCH_OPS_PER_WORKER = 10
_GOOGLE_OPS_PER_WORKER = 5  # each op costs a Vertex AI embedding call
_SESSION_OPS_PER_WORKER = 5
_MEMORY_OPS_PER_WORKER = 3
_MEMORY_POOL_SIZE = 20  # pre-created sessions cycled by memory ops

_QUERIES = [
    "cordless robot vacuum for pet hair",
    "espresso coffee machine with milk frother",
    "lightweight hiking backpack with rain cover",
    "wireless headphones with noise cancellation",
    "durable chew toys for dogs",
    "steam mop for hardwood floors",
    "air fryer with cooking presets",
    "waterproof tent for four people",
    "smart speaker with voice assistant",
    "cat tree with scratching posts",
]


@dataclass
class LevelStats:
  """Aggregated results for one scenario at one concurrency level."""

  concurrency: int
  ops: int = 0
  errors: list[str] = field(default_factory=list)
  latencies_ms: list[float] = field(default_factory=list)
  wall_seconds: float = 0.0

  def _percentile(self, p: float) -> float:
    ordered = sorted(self.latencies_ms)
    if not ordered:
      return 0.0
    rank = max(1, round(p / 100 * len(ordered))) - 1
    return ordered[min(rank, len(ordered) - 1)]

  def as_row(self) -> dict:
    lat = self.latencies_ms
    return {
        "concurrency": self.concurrency,
        "ops": self.ops,
        "errors": len(self.errors),
        "ops_per_sec": round(self.ops / self.wall_seconds, 1),
        "mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        "p50_ms": round(self._percentile(50), 1),
        "p95_ms": round(self._percentile(95), 1),
        "p99_ms": round(self._percentile(99), 1),
        "max_ms": round(max(lat), 1) if lat else None,
    }


async def _run_level(level: int, total_ops: int, op, label: str) -> LevelStats:
  """Runs total_ops of `op(i)` with at most `level` in flight."""
  semaphore = asyncio.Semaphore(level)
  stats = LevelStats(concurrency=level)

  async def worker(i: int) -> None:
    async with semaphore:
      start = time.perf_counter()
      try:
        await op(i)
      except Exception as exc:  # counted, not fatal: this is a load test.
        stats.errors.append(f"{type(exc).__name__}: {exc}")
      finally:
        stats.latencies_ms.append((time.perf_counter() - start) * 1000)
        stats.ops += 1

  wall_start = time.perf_counter()
  await asyncio.gather(*(worker(i) for i in range(total_ops)))
  stats.wall_seconds = time.perf_counter() - wall_start
  row = stats.as_row()
  print(
      f"    c={level:>2} ops={row['ops']:>3} errors={row['errors']}"
      f" p50={row['p50_ms']:>7}ms p95={row['p95_ms']:>7}ms"
      f" p99={row['p99_ms']:>7}ms throughput={row['ops_per_sec']:>6} ops/s"
  )
  if stats.errors:
    print(f"      first error: {stats.errors[0][:160]}")
  return stats


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    sys.exit(f"Environment variable {name} is required. See .env.")
  return value


def _baseline_rtt_ms(client: MongoClient, samples: int = 20) -> float:
  """Median round-trip time of a no-op ping, the latency floor."""
  latencies = []
  for _ in range(samples):
    start = time.perf_counter()
    client.admin.command("ping")
    latencies.append((time.perf_counter() - start) * 1000)
  return statistics.median(latencies)


async def main() -> None:
  uri = _require_env("MONGODB_URI")
  database = _require_env("MONGODB_DATABASE")
  auto_collection = _require_env("MONGODB_COLLECTION")
  google_collection = os.environ.get(
      "MONGODB_GOOGLE_COLLECTION", "products_google"
  )

  client = MongoClient(uri)
  client.admin.command("ping")
  build_info = client.admin.command("buildInfo")
  server_version = build_info.get("version", "unknown")
  baseline_ms = _baseline_rtt_ms(client)
  print(
      f"Connected. Server {server_version}, baseline RTT {baseline_ms:.1f} ms"
  )

  results: dict[str, list[dict]] = {}

  async def vector_auto(i: int) -> None:
    result = await _search_tool.vector_search(
        collection_name=auto_collection,
        query=_QUERIES[i % len(_QUERIES)],
        client=client,
        database_name=database,
        settings=_AUTO_SETTINGS,
        limit=4,
    )
    assert result["status"] == "SUCCESS", result

  async def hybrid_auto(i: int) -> None:
    result = await _search_tool.hybrid_search(
        collection_name=auto_collection,
        query=_QUERIES[i % len(_QUERIES)],
        text_search_field="description",
        client=client,
        database_name=database,
        settings=_AUTO_SETTINGS,
        limit=4,
    )
    assert result["status"] == "SUCCESS", result

  search_scenarios = [
      (
          "vector_search (Atlas auto-embedding)",
          vector_auto,
          _SEARCH_OPS_PER_WORKER,
      ),
      (
          "hybrid_search (Atlas auto-embedding)",
          hybrid_auto,
          _SEARCH_OPS_PER_WORKER,
      ),
  ]

  genai_client = _make_genai_client()
  if genai_client is not None:

    async def vector_google(i: int) -> None:
      result = await _search_tool.vector_search(
          collection_name=google_collection,
          query=_QUERIES[i % len(_QUERIES)],
          client=client,
          database_name=database,
          settings=_GOOGLE_SETTINGS,
          genai_client=genai_client,
          limit=4,
      )
      assert result["status"] == "SUCCESS", result

    search_scenarios.append((
        "vector_search (Google embedding + Vertex round trip)",
        vector_google,
        _GOOGLE_OPS_PER_WORKER,
    ))
  else:
    print("  [SKIP] Google embedding scenario: no Google credentials.")

  for label, op, ops_per_worker in search_scenarios:
    print(f"\n== {label} ==")
    await op(0)  # warm up: connection pools and index caches
    results[label] = []
    for level in _CONCURRENCY_LEVELS:
      stats = await _run_level(level, level * ops_per_worker, op, label)
      results[label].append(stats.as_row())

  # -- MongoDbSessionService: full write/read round trip per operation.
  print("\n== session service round trip (create+append+get+delete) ==")
  session_service = MongoDbSessionService(
      connection_string=uri, database_name=database
  )

  async def session_round_trip(i: int) -> None:
    user = f"lt_user_{i}"
    session = await session_service.create_session(
        app_name=_APP, user_id=user, state={"turn": 0}
    )
    event = Event(
        invocation_id=f"lt-{i}",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(text=f"load test message {i}")]
        ),
        timestamp=time.time(),
    )
    await session_service.append_event(session, event)
    fetched = await session_service.get_session(
        app_name=_APP, user_id=user, session_id=session.id
    )
    assert fetched is not None and len(fetched.events) == 1
    await session_service.delete_session(
        app_name=_APP, user_id=user, session_id=session.id
    )

  results["session_service round trip"] = []
  for level in _CONCURRENCY_LEVELS:
    stats = await _run_level(
        level,
        level * _SESSION_OPS_PER_WORKER,
        session_round_trip,
        "session",
    )
    results["session_service round trip"].append(stats.as_row())

  # -- MongoDbMemoryService: ingest (idempotent upsert) + keyword recall.
  print("\n== memory service (ingest + recall) ==")
  memory_service = MongoDbMemoryService(
      connection_string=uri, database_name=database
  )
  # Pre-create a pool of sessions to ingest; setup cost is not timed.
  pool = []
  for i in range(_MEMORY_POOL_SIZE):
    session = await session_service.create_session(
        app_name=_APP, user_id=f"lt_mem_user_{i % 5}"
    )
    await session_service.append_event(
        session,
        Event(
            invocation_id=f"lt-mem-{i}",
            author="user",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            f"load test memory {i} about"
                            f" {_QUERIES[i % len(_QUERIES)]}"
                        )
                    )
                ],
            ),
            timestamp=time.time(),
        ),
    )
    pool.append(session)

  async def memory_op(i: int) -> None:
    session = pool[i % _MEMORY_POOL_SIZE]
    reloaded = await session_service.get_session(
        app_name=_APP, user_id=session.user_id, session_id=session.id
    )
    await memory_service.add_session_to_memory(reloaded)
    response = await memory_service.search_memory(
        app_name=_APP,
        user_id=session.user_id,
        query=_QUERIES[i % len(_QUERIES)],
    )
    assert response.memories is not None

  results["memory_service ingest + recall"] = []
  for level in _CONCURRENCY_LEVELS:
    stats = await _run_level(
        level, level * _MEMORY_OPS_PER_WORKER, memory_op, "memory"
    )
    results["memory_service ingest + recall"].append(stats.as_row())

  # Cleanup: every document created under the load-test app name.
  db = client[database]
  db["events"].delete_many({"app_name": _APP})
  db["sessions"].delete_many({"_id": {"$regex": f"^{_APP}/"}})
  db["app_states"].delete_many({"_id": _APP})
  db["user_states"].delete_many({"_id": {"$regex": f"^{_APP}/"}})
  db["memories"].delete_many({"app_name": _APP})
  await session_service.close()
  await memory_service.close()
  client.close()
  print("\nCleanup done: all load-test documents removed.")

  _write_report(results, server_version, baseline_ms)


def _write_report(
    results: dict[str, list[dict]], server_version: str, baseline_ms: float
) -> None:
  """Writes load_test_report.md and load_test_results.json."""
  lines = [
      "# MongoDB Integration — Load Test Report",
      "",
      f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
      (
          f"- **Host:** {getpass.getuser()}@{platform.node()}"
          f" ({platform.system()} {platform.release()})"
      ),
      f"- **Python:** {platform.python_version()}",
      f"- **MongoDB server:** {server_version}",
      (
          f"- **Network baseline (median ping RTT):** {baseline_ms:.1f} ms —"
          " every operation sits on top of this floor"
      ),
      f"- **Concurrency levels:** {_CONCURRENCY_LEVELS}",
      "",
      "## Method",
      "",
      "Each scenario runs `level x ops-per-worker` operations with at most",
      "`level` in flight (asyncio semaphore). Latency is per-operation wall",
      "time from this client, so it includes the network round trip to Atlas",
      "and, where noted, server-side Voyage AI embedding (auto-embedding",
      "mode) or a client-side Vertex AI embedding call (Google mode). The",
      "session round trip bundles four operations (create + append + get +",
      "delete); the memory scenario bundles ingest (idempotent upsert over a",
      "pre-created pool) + keyword recall. All state writes used a dedicated",
      "app name and were cleaned up after the run.",
      "",
      "## Results",
  ]
  for scenario, rows in results.items():
    lines += [
        "",
        f"### {scenario}",
        "",
        (
            "| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) |"
            " p95 (ms) | p99 (ms) | Max (ms) |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
      lines.append(
          f"| {row['concurrency']} | {row['ops']} | {row['errors']} |"
          f" {row['ops_per_sec']} | {row['mean_ms']} | {row['p50_ms']} |"
          f" {row['p95_ms']} | {row['p99_ms']} | {row['max_ms']} |"
      )

  lines += ["", "## Observations", ""]
  for scenario, rows in results.items():
    if not rows or not rows[0]["p50_ms"]:
      continue
    low, high = rows[0], rows[-1]
    scaling = (
        high["ops_per_sec"] / low["ops_per_sec"] if low["ops_per_sec"] else 0
    )
    total_errors = sum(row["errors"] for row in rows)
    lines.append(
        f"- **{scenario}:** p50 {low['p50_ms']} ms at concurrency 1 grows to"
        f" {high['p50_ms']} ms at concurrency {high['concurrency']};"
        f" throughput scales {scaling:.1f}x"
        f" ({low['ops_per_sec']} -> {high['ops_per_sec']} ops/s);"
        f" {total_errors} errors across {sum(r['ops'] for r in rows)} ops."
    )
  lines += [
      "",
      "Throughput scaling below the concurrency increase indicates a",
      "saturation point — client CPU, the asyncio executor, the Atlas tier,",
      "or (Google mode) the embedding API round trip. Compare p50 against",
      "the network baseline to separate server work from transit time.",
      "",
  ]

  out_dir = Path(__file__).resolve().parent
  report_path = out_dir / "load_test_report.md"
  report_path.write_text("\n".join(lines), encoding="utf-8")
  json_path = out_dir / "load_test_results.json"
  json_path.write_text(
      json.dumps(
          {
              "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "server_version": server_version,
              "baseline_rtt_ms": baseline_ms,
              "results": results,
          },
          indent=2,
      ),
      encoding="utf-8",
  )
  print(f"\nReport written to {report_path}")
  print(f"Raw results written to {json_path}")


if __name__ == "__main__":
  asyncio.run(main())
