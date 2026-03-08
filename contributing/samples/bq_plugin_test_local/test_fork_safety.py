"""Test fork-safety fix for issue #4636.

Verifies that the PID-tracking fix prevents gRPC deadlocks when the
BigQueryAgentAnalyticsPlugin is used across process boundaries.

Usage:
  python contributing/samples/bq_plugin_test_local/test_fork_safety.py

This script runs three tests:

Test 1 - PID mismatch detection (simulated fork):
  Manipulates _init_pid to simulate what os.fork() does, then verifies
  _ensure_started() detects the mismatch and re-initializes cleanly.

Test 2 - Spawn-based multiprocessing (safe path):
  Uses ProcessPoolExecutor with 'spawn' start method. The plugin is
  pickled via __getstate__/__setstate__, which resets _init_pid=0 and
  _started=False. The child process re-initializes from scratch.
  Verifies events are logged from both parent and child.

Test 3 - Sequential re-initialization:
  Starts the plugin, then resets it (simulating fork), then starts
  again. Verifies the second initialization works and events are logged.
"""

import asyncio
import concurrent.futures
import multiprocessing
import os
import sys
import time

# Configure Vertex AI backend before any ADK imports.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-0728-467323"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

import random

from google.adk.agents.llm_agent import LlmAgent
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


def roll_die(sides: int) -> int:
  """Roll a die and return the result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def create_plugin():
  """Create a fresh plugin instance."""
  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      create_views=False,
  )
  return BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )


async def run_agent_with_plugin(plugin, label):
  """Run the agent with the given plugin and return the session ID."""
  pid = os.getpid()
  print(f"  [{label}] PID={pid} starting agent run...")

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name=f"fork_test_{label}",
      description="Fork safety test agent.",
      instruction="You are a helpful assistant. Use tools when asked.",
      tools=[roll_die],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=f"fork_test_{label}",
      agent=agent,
      session_service=session_service,
      plugins=[plugin],
  )

  session = await session_service.create_session(
      app_name=f"fork_test_{label}",
      user_id=f"test-user-{label}",
  )

  query = f"Roll a 20-sided die (from {label}, PID {pid})"
  content = types.Content(role="user", parts=[types.Part(text=query)])
  response_text = ""
  async for event in runner.run_async(
      user_id=f"test-user-{label}",
      session_id=session.id,
      new_message=content,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          response_text += part.text

  print(f"  [{label}] Response: {response_text[:120]}")
  await plugin.flush()
  await asyncio.sleep(1)
  return session.id


def worker_run_agent(label):
  """Worker function for ProcessPoolExecutor (spawn mode)."""
  # In 'spawn' mode, this is a fresh process. The plugin module is
  # re-imported, but we create a new plugin to avoid sharing state.
  plugin = create_plugin()
  sid = asyncio.run(run_agent_with_plugin(plugin, label))
  asyncio.run(plugin.shutdown())
  return sid


async def test1_pid_mismatch_detection():
  """Test 1: Simulated fork via PID manipulation."""
  print("\n" + "=" * 60)
  print("TEST 1: PID Mismatch Detection (Simulated Fork)")
  print("=" * 60)

  plugin = create_plugin()

  # Initialize in "parent"
  print("\n  Initializing plugin in parent...")
  await plugin._ensure_started()
  assert plugin._started, "Plugin should be started"
  original_pid = plugin._init_pid
  original_client = plugin.client
  print(f"  Parent PID: {original_pid}")
  print(f"  Plugin._started: {plugin._started}")
  print(f"  Plugin.client id: {id(original_client)}")

  # Simulate fork: child inherits everything but has different PID
  print("\n  Simulating fork (setting _init_pid to fake value)...")
  plugin._init_pid = original_pid + 99999  # Fake "parent" PID

  # Now _ensure_started should detect PID mismatch and reset
  print("  Calling _ensure_started() in 'child'...")
  await plugin._ensure_started()

  new_client = plugin.client
  new_pid = plugin._init_pid

  print(f"  New PID recorded: {new_pid}")
  print(f"  New client id: {id(new_client)}")
  print(f"  Plugin._started: {plugin._started}")

  # Verify re-initialization happened
  assert (
      new_pid == os.getpid()
  ), f"_init_pid should be current PID ({os.getpid()}), got {new_pid}"
  assert (
      new_client is not original_client
  ), "Client should be a new instance after fork detection"
  assert plugin._started, "Plugin should be started after re-init"

  # Actually use the re-initialized plugin
  print("\n  Running agent with re-initialized plugin...")
  sid = await run_agent_with_plugin(plugin, "simulated_child")
  await plugin.shutdown()

  print(
      "\n  TEST 1 PASSED: PID mismatch detected, re-initialized,"
      f" and logged events (session={sid[:12]}...)"
  )
  return sid


async def test2_spawn_multiprocessing():
  """Test 2: Real multiprocessing with 'spawn' method."""
  print("\n" + "=" * 60)
  print("TEST 2: Spawn-based Multiprocessing")
  print("=" * 60)

  # Ensure spawn method
  try:
    ctx = multiprocessing.get_context("spawn")
  except ValueError:
    print("  SKIP: 'spawn' method not available")
    return []

  # Run in parent first
  parent_plugin = create_plugin()
  print("\n  Running agent in PARENT process...")
  parent_sid = await run_agent_with_plugin(parent_plugin, "parent")
  print(f"  Parent plugin._started: {parent_plugin._started}")
  print(f"  Parent plugin._init_pid: {parent_plugin._init_pid}")
  await parent_plugin.shutdown()

  # Run in child processes via spawn
  print("\n  Spawning child processes...")
  child_sids = []

  # Use spawn context explicitly
  with concurrent.futures.ProcessPoolExecutor(
      max_workers=2, mp_context=ctx
  ) as executor:
    futures = {
        executor.submit(worker_run_agent, f"spawn_child_{i}"): i
        for i in range(2)
    }
    for future in concurrent.futures.as_completed(futures, timeout=120):
      idx = futures[future]
      try:
        sid = future.result(timeout=60)
        print(f"  Child {idx} completed. Session: {sid[:12]}...")
        child_sids.append(sid)
      except Exception as e:
        print(f"  Child {idx} FAILED: {type(e).__name__}: {e}")

  all_sids = [parent_sid] + child_sids
  if len(child_sids) == 2:
    print(f"\n  TEST 2 PASSED: Parent + 2 children ran successfully.")
  else:
    print(f"\n  TEST 2 FAILED: Expected 2 children, got {len(child_sids)}")
  return all_sids


async def test3_sequential_reinit():
  """Test 3: Reset and re-initialize (simulates fork recovery)."""
  print("\n" + "=" * 60)
  print("TEST 3: Sequential Re-initialization")
  print("=" * 60)

  plugin = create_plugin()

  # First initialization
  print("\n  First initialization...")
  sid1 = await run_agent_with_plugin(plugin, "init1")
  print(f"  Session 1: {sid1[:12]}...")

  # Simulate fork and reset
  print("\n  Simulating fork + reset...")
  plugin._reset_runtime_state()
  assert not plugin._started, "Should be reset"
  assert plugin.client is None, "Client should be None"
  assert plugin._init_pid == os.getpid(), "PID should be current"
  print(f"  After reset: _started={plugin._started}, client={plugin.client}")

  # Second initialization
  print("\n  Second initialization (re-init after reset)...")
  sid2 = await run_agent_with_plugin(plugin, "init2")
  print(f"  Session 2: {sid2[:12]}...")

  await plugin.shutdown()

  print(
      f"\n  TEST 3 PASSED: Plugin re-initialized and logged events"
      f" in both sessions."
  )
  return [sid1, sid2]


def verify_all_events(session_ids):
  """Verify events exist in BigQuery for all sessions."""
  print("\n" + "=" * 60)
  print("VERIFICATION: Checking BigQuery for all sessions")
  print("=" * 60)

  client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
  full_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

  all_ok = True
  total_events = 0

  for sid in session_ids:
    query = f"""
    SELECT COUNT(*) as total
    FROM `{full_table}`
    WHERE session_id = '{sid}'
    """
    result = client.query(query).result()
    count = list(result)[0].total
    total_events += count
    status = "OK" if count > 0 else "MISSING"
    print(f"  Session {sid[:12]}... -> {count} events [{status}]")
    if count == 0:
      all_ok = False

  print(f"\n  Total events across all sessions: {total_events}")
  return all_ok


async def main():
  print("=" * 60)
  print("Fork-Safety Test Suite (Issue #4636)")
  print(f"PID: {os.getpid()}, Platform: {sys.platform}")
  print("=" * 60)

  all_session_ids = []

  # Test 1: Simulated fork
  sid = await test1_pid_mismatch_detection()
  all_session_ids.append(sid)

  # Test 2: Real multiprocessing with spawn
  sids = await test2_spawn_multiprocessing()
  all_session_ids.extend(sids)

  # Test 3: Sequential re-init
  sids = await test3_sequential_reinit()
  all_session_ids.extend(sids)

  # Wait for BQ data to settle
  print("\nWaiting 5s for BigQuery data to settle...")
  time.sleep(5)

  # Verify all events
  success = verify_all_events(all_session_ids)

  if success:
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(
        "Fork-safety fix (issue #4636) verified with"
        f" {len(all_session_ids)} sessions."
    )
    print("=" * 60)
    return 0
  else:
    print("\n" + "=" * 60)
    print("SOME TESTS FAILED")
    print("=" * 60)
    return 1


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
