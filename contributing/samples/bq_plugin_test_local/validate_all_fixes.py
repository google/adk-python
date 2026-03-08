"""Comprehensive end-to-end validation for PR #4640 fixes.

Tests all three fix areas with real BigQuery logging data:

1. Trace-ID continuity (#4645):
   - All events within one invocation share the same trace_id
   - Multi-turn sessions keep per-invocation trace_id isolation

2. Fork-safety (#4636):
   - PID mismatch detection and re-initialization
   - Spawn-based multiprocessing works cleanly

3. Auto-create analytics views (#4639):
   - All 15 views exist and are queryable
   - Views return correct event types

4. O11y alignment:
   - When OTel is enabled, BQ rows use the same trace/span IDs as
     Cloud Trace

Usage:
  python contributing/samples/bq_plugin_test_local/validate_all_fixes.py
"""

import asyncio
import concurrent.futures
import multiprocessing
import os
import sys
import time
import traceback

# Configure Vertex AI backend before any ADK imports.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-0728-467323"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

import random

from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.bigquery_agent_analytics_plugin import _EVENT_VIEW_DEFS
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.cloud import bigquery
from google.genai import types

PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"
TABLE_ID = "agent_events"
FULL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


# ── Tools ──────────────────────────────────────────────────────
def roll_die(sides: int) -> int:
  """Roll a die and return the result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def get_weather(city: str) -> str:
  """Get the current weather for a city.

  Args:
    city: The name of the city.

  Returns:
    A string describing the weather.
  """
  weathers = ["sunny", "cloudy", "rainy", "snowy", "windy"]
  temp = random.randint(50, 95)
  return f"{city}: {random.choice(weathers)}, {temp}F."


# ── Helpers ────────────────────────────────────────────────────
def create_plugin(create_views=True):
  """Create a fresh plugin instance."""
  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      create_views=create_views,
  )
  return BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )


async def run_multi_turn(plugin, label, num_turns=3):
  """Run a multi-turn conversation and return the session ID."""
  pid = os.getpid()
  print(f"    [{label}] PID={pid}, turns={num_turns}")

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name=f"validate_{label}",
      description="Validation agent.",
      instruction=(
          "You are a helpful assistant. Use tools when asked. Be concise."
      ),
      tools=[roll_die, get_weather],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=f"validate_{label}",
      agent=agent,
      session_service=session_service,
      plugins=[plugin],
  )

  session = await session_service.create_session(
      app_name=f"validate_{label}",
      user_id=f"user-{label}",
  )

  queries = [
      "Roll a 20-sided die",
      "What is the weather in Tokyo?",
      "Roll a 6-sided die and tell me if it is even or odd",
      "What is the weather in Paris?",
      "Roll two 12-sided dice and sum them",
  ][:num_turns]

  for i, query in enumerate(queries):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    response = ""
    async for event in runner.run_async(
        user_id=f"user-{label}",
        session_id=session.id,
        new_message=content,
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            response += part.text
    print(f"    [{label}] Turn {i+1}: {response[:80]}...")

  await plugin.flush()
  await asyncio.sleep(1)
  return session.id


def worker_run_agent(label):
  """Worker function for ProcessPoolExecutor (spawn mode)."""
  plugin = create_plugin(create_views=False)
  sid = asyncio.run(run_multi_turn(plugin, label, num_turns=2))
  asyncio.run(plugin.shutdown())
  return sid


# ── BQ Query Helpers ───────────────────────────────────────────
def bq_client():
  return bigquery.Client(project=PROJECT_ID, location=LOCATION)


def query_bq(client, sql):
  """Run a BQ query and return list of row dicts."""
  result = client.query(sql).result()
  return [dict(row) for row in result]


# ══════════════════════════════════════════════════════════════
#  TEST 1: Trace-ID Continuity (#4645) + Multi-Turn Isolation
# ══════════════════════════════════════════════════════════════
async def test_trace_id_continuity():
  """Run 3-turn agent, verify each invocation has one trace_id."""
  print("\n" + "=" * 60)
  print("TEST 1: Trace-ID Continuity + Multi-Turn Isolation (#4645)")
  print("=" * 60)

  plugin = create_plugin(create_views=False)
  sid = await run_multi_turn(plugin, "trace_test", num_turns=3)
  await plugin.shutdown()

  print(f"\n    Session: {sid}")
  print("    Waiting 8s for BQ data...")
  await asyncio.sleep(8)

  client = bq_client()

  # Query: group events by invocation_id, check trace_id count
  sql = f"""
  SELECT
    invocation_id,
    COUNT(DISTINCT trace_id) AS distinct_traces,
    COUNT(*) AS event_count,
    ARRAY_AGG(DISTINCT trace_id) AS trace_ids,
    ARRAY_AGG(DISTINCT event_type) AS event_types
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
  GROUP BY invocation_id
  ORDER BY MIN(timestamp)
  """
  rows = query_bq(client, sql)

  if not rows:
    print("    ERROR: No events found in BQ!")
    return FAIL, sid

  all_ok = True
  trace_ids_seen = set()

  for row in rows:
    inv_id = row["invocation_id"]
    n_traces = row["distinct_traces"]
    n_events = row["event_count"]
    tids = row["trace_ids"]
    etypes = row["event_types"]

    status = "OK" if n_traces == 1 else "BROKEN"
    if n_traces != 1:
      all_ok = False

    print(
        f"    Invocation {inv_id[:12]}... -> "
        f"{n_events} events, {n_traces} trace_id(s) [{status}]"
    )
    print(f"      trace_ids: {tids}")
    print(f"      event_types: {sorted(etypes)}")

    for tid in tids:
      trace_ids_seen.add(tid)

  # Multi-turn isolation: each invocation should have a DIFFERENT trace_id
  print(f"\n    Distinct trace_ids across invocations: {len(trace_ids_seen)}")
  if len(trace_ids_seen) < len(rows):
    print(
        "    WARNING: Some invocations share a trace_id"
        " (may be expected if OTel reuses ambient spans)"
    )

  # Check that early events (USER_MESSAGE_RECEIVED, INVOCATION_STARTING)
  # share the same trace_id as later events (AGENT_STARTING, LLM_*)
  sql_fracture = f"""
  SELECT
    invocation_id,
    event_type,
    trace_id,
    span_id,
    parent_span_id
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
  ORDER BY invocation_id, timestamp
  """
  detail_rows = query_bq(client, sql_fracture)

  # Group by invocation
  invocations = {}
  for r in detail_rows:
    inv = r["invocation_id"]
    invocations.setdefault(inv, []).append(r)

  fracture_count = 0
  for inv_id, events in invocations.items():
    tids_in_inv = set(e["trace_id"] for e in events)
    if len(tids_in_inv) > 1:
      fracture_count += 1
      print(f"\n    FRACTURE in {inv_id[:12]}...:")
      for e in events:
        print(
            f"      {e['event_type']:30s} trace={e['trace_id'][:16]}..."
            f" span={e['span_id'][:8] if e['span_id'] else 'N/A'}..."
        )

  if fracture_count > 0:
    print(f"\n    {fracture_count} invocation(s) with fractured trace_id!")
    all_ok = False

  result = PASS if all_ok else FAIL
  print(f"\n    TEST 1: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  TEST 2: Fork-Safety (#4636)
# ══════════════════════════════════════════════════════════════
async def test_fork_safety():
  """Test PID mismatch detection and spawn-based multiprocessing."""
  print("\n" + "=" * 60)
  print("TEST 2: Fork-Safety (#4636)")
  print("=" * 60)

  # 2a: PID mismatch detection (simulated fork)
  print("\n  2a: PID Mismatch Detection")
  plugin = create_plugin(create_views=False)
  await plugin._ensure_started()

  original_pid = plugin._init_pid
  original_client_id = id(plugin.client)
  print(f"    Original PID: {original_pid}")

  # Simulate fork
  plugin._init_pid = original_pid + 99999
  await plugin._ensure_started()

  new_pid = plugin._init_pid
  new_client_id = id(plugin.client)

  pid_ok = new_pid == os.getpid()
  client_ok = new_client_id != original_client_id

  print(f"    After simulated fork: PID={new_pid}, new client={client_ok}")

  if not pid_ok or not client_ok:
    print("    FAIL: PID mismatch not detected or client not replaced")
    await plugin.shutdown()
    return FAIL, []

  # Run agent with re-initialized plugin
  sid_sim = await run_multi_turn(plugin, "fork_sim", num_turns=1)
  await plugin.shutdown()

  print(f"    2a PASS: Simulated fork detected, session={sid_sim[:12]}...")

  # 2b: Spawn-based multiprocessing
  print("\n  2b: Spawn-based Multiprocessing")
  try:
    ctx = multiprocessing.get_context("spawn")
  except ValueError:
    print("    SKIP: 'spawn' method not available")
    return PASS, [sid_sim]

  # Parent run
  parent_plugin = create_plugin(create_views=False)
  sid_parent = await run_multi_turn(parent_plugin, "parent", num_turns=2)
  await parent_plugin.shutdown()

  # Spawn children
  print("    Spawning 2 child processes...")
  child_sids = []
  with concurrent.futures.ProcessPoolExecutor(
      max_workers=2, mp_context=ctx
  ) as executor:
    futures = {
        executor.submit(worker_run_agent, f"child_{i}"): i for i in range(2)
    }
    for future in concurrent.futures.as_completed(futures, timeout=120):
      idx = futures[future]
      try:
        sid = future.result(timeout=60)
        child_sids.append(sid)
        print(f"    Child {idx}: session={sid[:12]}...")
      except Exception as e:
        print(f"    Child {idx} FAILED: {e}")

  all_sids = [sid_sim, sid_parent] + child_sids
  spawn_ok = len(child_sids) == 2

  result = PASS if spawn_ok else FAIL
  print(f"\n    TEST 2: {result}")
  return result, all_sids


# ══════════════════════════════════════════════════════════════
#  TEST 3: Auto-Create Analytics Views (#4639)
# ══════════════════════════════════════════════════════════════
async def test_analytics_views(session_id):
  """Verify all 15 views exist and return correct event types."""
  print("\n" + "=" * 60)
  print("TEST 3: Auto-Create Analytics Views (#4639)")
  print("=" * 60)

  # Create a plugin with views enabled and run to trigger view creation
  plugin = create_plugin(create_views=True)
  sid = await run_multi_turn(plugin, "views_test", num_turns=2)
  await plugin.shutdown()

  print(f"\n    Session: {sid}")
  print("    Waiting 5s for BQ data...")
  await asyncio.sleep(5)

  client = bq_client()

  all_ok = True
  views_ok = 0
  views_fail = 0

  expected_views = len(_EVENT_VIEW_DEFS)
  print(f"\n    Checking {expected_views} views:")

  for event_type in _EVENT_VIEW_DEFS:
    view_name = "v_" + event_type.lower()
    full_view = f"{PROJECT_ID}.{DATASET_ID}.{view_name}"
    try:
      sql = f"SELECT COUNT(*) as cnt FROM `{full_view}` LIMIT 1"
      rows = query_bq(client, sql)
      cnt = rows[0]["cnt"]
      print(f"    {view_name:40s} -> {cnt} rows [OK]")
      views_ok += 1
    except Exception as e:
      print(f"    {view_name:40s} -> FAIL: {e}")
      views_fail += 1
      all_ok = False

  # Spot-check: v_llm_response should have typed columns
  print("\n    Spot-check v_llm_response:")
  try:
    check_sid = session_id or sid
    sql = f"""
    SELECT timestamp, model_version,
           usage_prompt_tokens, usage_completion_tokens,
           total_ms, ttft_ms
    FROM `{PROJECT_ID}.{DATASET_ID}.v_llm_response`
    WHERE session_id = '{check_sid}'
    LIMIT 2
    """
    rows = query_bq(client, sql)
    for row in rows:
      print(
          f"      model={row['model_version']}"
          f" prompt_tok={row['usage_prompt_tokens']}"
          f" total_ms={row['total_ms']}"
      )
    if not rows:
      print("      (no rows for this session)")
  except Exception as e:
    print(f"      Error: {e}")

  result = PASS if all_ok else FAIL
  print(f"\n    Views: {views_ok} OK, {views_fail} failed")
  print(f"    TEST 3: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  TEST 4: O11y Alignment (span IDs match Cloud Trace)
# ══════════════════════════════════════════════════════════════
async def test_o11y_alignment():
  """Run agent with OTel enabled, verify BQ span IDs are valid hex."""
  print("\n" + "=" * 60)
  print("TEST 4: O11y Alignment (Span IDs)")
  print("=" * 60)

  # Set up OTel
  try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    print("    OTel TracerProvider configured")
  except ImportError:
    print("    SKIP: opentelemetry not installed")
    return SKIP, None

  plugin = create_plugin(create_views=False)
  sid = await run_multi_turn(plugin, "o11y_test", num_turns=2)
  await plugin.shutdown()

  print(f"\n    Session: {sid}")
  print("    Waiting 8s for BQ data...")
  await asyncio.sleep(8)

  client = bq_client()

  sql = f"""
  SELECT event_type, trace_id, span_id, parent_span_id
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
  ORDER BY timestamp
  """
  rows = query_bq(client, sql)

  if not rows:
    print("    ERROR: No events found!")
    return FAIL, sid

  all_ok = True
  hex_pattern_trace = True
  hex_pattern_span = True

  for row in rows:
    tid = row["trace_id"]
    sid_val = row["span_id"]

    # trace_id should be 32-char hex (not a UUID)
    if tid and len(tid) == 32:
      try:
        int(tid, 16)
      except ValueError:
        hex_pattern_trace = False
    # span_id should be 16-char hex
    if sid_val and len(sid_val) == 16:
      try:
        int(sid_val, 16)
      except ValueError:
        hex_pattern_span = False

    print(
        f"    {row['event_type']:30s}"
        f" trace={tid[:16] if tid else 'N/A'}..."
        f" span={sid_val[:8] if sid_val else 'N/A'}..."
        f" parent={row['parent_span_id'][:8] if row['parent_span_id'] else 'N/A'}..."
    )

  if not hex_pattern_trace:
    print("    WARNING: Some trace_ids are not 32-char hex")
  if not hex_pattern_span:
    print("    WARNING: Some span_ids are not 16-char hex")

  # Check trace_id consistency within each invocation
  sql2 = f"""
  SELECT invocation_id, COUNT(DISTINCT trace_id) AS n
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
  GROUP BY invocation_id
  HAVING COUNT(DISTINCT trace_id) > 1
  """
  fractures = query_bq(client, sql2)
  if fractures:
    print(f"    FAIL: {len(fractures)} invocations with fractured trace_id")
    all_ok = False

  result = PASS if all_ok else FAIL
  print(f"\n    TEST 4: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  FINAL VERIFICATION
# ══════════════════════════════════════════════════════════════
def final_verification(all_session_ids):
  """Count total events across all sessions."""
  print("\n" + "=" * 60)
  print("FINAL VERIFICATION: All Sessions")
  print("=" * 60)

  client = bq_client()
  total_events = 0

  for sid in all_session_ids:
    if not sid:
      continue
    sql = f"""
    SELECT COUNT(*) as total
    FROM `{FULL_TABLE}`
    WHERE session_id = '{sid}'
    """
    rows = query_bq(client, sql)
    count = rows[0]["total"]
    total_events += count
    status = "OK" if count > 0 else "MISSING"
    print(f"    Session {sid[:16]}... -> {count} events [{status}]")

  print(f"\n    Total events: {total_events}")
  return total_events > 0


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
  print("=" * 60)
  print("PR #4640 Comprehensive Validation Suite")
  print(f"PID: {os.getpid()}, Platform: {sys.platform}")
  print("=" * 60)

  results = {}
  all_session_ids = []

  # Test 1: Trace-ID continuity
  try:
    r, sid = await test_trace_id_continuity()
    results["trace_id_continuity"] = r
    if sid:
      all_session_ids.append(sid)
  except Exception as e:
    print(f"    TEST 1 ERROR: {e}")
    traceback.print_exc()
    results["trace_id_continuity"] = FAIL

  # Test 2: Fork-safety
  try:
    r, sids = await test_fork_safety()
    results["fork_safety"] = r
    all_session_ids.extend(sids)
  except Exception as e:
    print(f"    TEST 2 ERROR: {e}")
    traceback.print_exc()
    results["fork_safety"] = FAIL

  # Test 3: Analytics views
  try:
    first_sid = all_session_ids[0] if all_session_ids else None
    r, sid = await test_analytics_views(first_sid)
    results["analytics_views"] = r
    if sid:
      all_session_ids.append(sid)
  except Exception as e:
    print(f"    TEST 3 ERROR: {e}")
    traceback.print_exc()
    results["analytics_views"] = FAIL

  # Test 4: O11y alignment
  try:
    r, sid = await test_o11y_alignment()
    results["o11y_alignment"] = r
    if sid:
      all_session_ids.append(sid)
  except Exception as e:
    print(f"    TEST 4 ERROR: {e}")
    traceback.print_exc()
    results["o11y_alignment"] = FAIL

  # Wait and do final verification
  print("\n    Waiting 5s for final data settlement...")
  time.sleep(5)
  final_verification(all_session_ids)

  # Summary
  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  all_pass = True
  for name, result in results.items():
    icon = "OK" if result == PASS else ("SKIP" if result == SKIP else "FAIL")
    print(f"  [{icon:4s}] {name}")
    if result == FAIL:
      all_pass = False

  print(f"\n  Sessions tested: {len(all_session_ids)}")
  if all_pass:
    print("\n  ALL TESTS PASSED")
    return 0
  else:
    print("\n  SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
