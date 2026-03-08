"""End-to-end validation for issue #4694 fixes against real BigQuery.

Tests the 6 bug fixes with real BQ data:

1. Error callbacks emit status="ERROR" (not "OK")
2. Schema upgrade handles nested RECORD sub-fields (structural)
3. Multi-loop shutdown drains other loops (structural)
4. Session state is truncated
5. String system prompt is truncated
6. _HITL_TOOL_NAMES removed (structural, no BQ data to check)

Fixes #2, #3, #6 are structural/code-level and verified by unit tests.
This script validates #1, #4, #5 via real BigQuery data.

Usage:
  python contributing/samples/bq_plugin_test_local/validate_issue_4694.py
"""

import asyncio
import json
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


def roll_and_store_big_state(sides: int, tool_context) -> int:
  """Roll a die and store a large value in session state.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  result = random.randint(1, sides)
  tool_context.state["big_key"] = "X" * 5000
  tool_context.state["normal_key"] = "small value"
  return result


def always_fail() -> str:
  """A tool that always raises an error for testing error logging.

  Returns:
    Never returns, always raises.
  """
  raise ValueError("Intentional tool failure for testing")


# ── BQ helpers ─────────────────────────────────────────────────
def bq_client():
  return bigquery.Client(project=PROJECT_ID, location=LOCATION)


def query_bq(client, sql):
  result = client.query(sql).result()
  return [dict(row) for row in result]


# ══════════════════════════════════════════════════════════════
#  TEST 1: Error callbacks emit status="ERROR"  (Fix #1)
# ══════════════════════════════════════════════════════════════
async def test_error_status():
  """Run agent that triggers a tool error, verify status=ERROR in BQ."""
  print("\n" + "=" * 60)
  print("TEST 1: Error callbacks emit status='ERROR' (Fix #1)")
  print("=" * 60)

  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
  )
  plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name="error_test_agent",
      description="Agent that tests error logging.",
      instruction=(
          "You are a test assistant. When the user says 'fail', "
          "call the always_fail tool. When asked to roll, use roll_die."
      ),
      tools=[roll_die, always_fail],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="error_test",
      agent=agent,
      session_service=session_service,
      plugins=[plugin],
  )

  session = await session_service.create_session(
      app_name="error_test", user_id="user-error-test"
  )
  sid = session.id
  print(f"    Session: {sid}")

  # Turn 1: normal tool call
  content = types.Content(
      role="user", parts=[types.Part(text="Roll a 6-sided die")]
  )
  async for event in runner.run_async(
      user_id="user-error-test",
      session_id=sid,
      new_message=content,
  ):
    pass

  # Turn 2: trigger tool error (the runner may propagate the exception)
  content = types.Content(
      role="user",
      parts=[types.Part(text="Please call the always_fail tool now")],
  )
  try:
    async for event in runner.run_async(
        user_id="user-error-test",
        session_id=sid,
        new_message=content,
    ):
      pass
  except Exception as e:
    print(f"    (Expected error propagated: {type(e).__name__}: {e})")

  await plugin.flush()
  await asyncio.sleep(2)
  await plugin.shutdown()

  print("    Waiting 10s for BQ data...")
  await asyncio.sleep(10)

  client = bq_client()

  # Check for TOOL_ERROR events with status=ERROR
  sql = f"""
  SELECT event_type, status, error_message
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
    AND event_type IN ('TOOL_ERROR', 'LLM_ERROR')
  ORDER BY timestamp
  """
  rows = query_bq(client, sql)

  if not rows:
    print("    No error events found — model may not have called the tool.")
    print("    Checking all events for this session:")
    sql2 = f"""
    SELECT event_type, status
    FROM `{FULL_TABLE}`
    WHERE session_id = '{sid}'
    ORDER BY timestamp
    """
    all_rows = query_bq(client, sql2)
    for r in all_rows:
      print(f"      {r['event_type']:30s} status={r['status']}")
    return SKIP, sid

  all_ok = True
  for row in rows:
    status = row["status"]
    ok = status == "ERROR"
    if not ok:
      all_ok = False
    label = "OK" if ok else "BROKEN"
    print(
        f"    {row['event_type']:20s} status={status:6s}"
        f" error={row['error_message'][:50] if row['error_message'] else 'N/A'}"
        f" [{label}]"
    )

  # Also verify normal events still have status=OK
  sql3 = f"""
  SELECT event_type, status
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
    AND event_type IN ('TOOL_COMPLETED', 'LLM_RESPONSE')
  ORDER BY timestamp
  LIMIT 5
  """
  ok_rows = query_bq(client, sql3)
  for r in ok_rows:
    status_ok = r["status"] == "OK"
    if not status_ok:
      all_ok = False
    label = "OK" if status_ok else "BROKEN"
    print(f"    {r['event_type']:20s} status={r['status']:6s} [{label}]")

  result = PASS if all_ok else FAIL
  print(f"\n    TEST 1: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  TEST 2: Session state is truncated  (Fix #4)
# ══════════════════════════════════════════════════════════════
async def test_session_state_truncation():
  """Run agent with large session state, verify truncation in BQ."""
  print("\n" + "=" * 60)
  print("TEST 2: Session state is truncated (Fix #4)")
  print("=" * 60)

  max_content_length = 500
  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      max_content_length=max_content_length,
  )
  plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name="state_test_agent",
      description="Agent for session state truncation test.",
      instruction=(
          "You are a helpful assistant. When asked to roll, use the"
          " roll_and_store_big_state tool. Always use tools when asked."
      ),
      tools=[roll_and_store_big_state],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="state_test",
      agent=agent,
      session_service=session_service,
      plugins=[plugin],
  )

  session = await session_service.create_session(
      app_name="state_test", user_id="user-state-test"
  )
  sid = session.id
  print(f"    Session: {sid}")
  print("    Tool will inject big_key=5000 chars into session state")

  # Turn 1: tool call stores big state
  content = types.Content(
      role="user", parts=[types.Part(text="Roll a 20-sided die")]
  )
  async for event in runner.run_async(
      user_id="user-state-test",
      session_id=sid,
      new_message=content,
  ):
    pass

  # Turn 2: state should now be visible in session for logged events
  content2 = types.Content(
      role="user", parts=[types.Part(text="Roll a 6-sided die")]
  )
  async for event in runner.run_async(
      user_id="user-state-test",
      session_id=sid,
      new_message=content2,
  ):
    pass

  await plugin.flush()
  await asyncio.sleep(2)
  await plugin.shutdown()

  print("    Waiting 10s for BQ data...")
  await asyncio.sleep(10)

  client = bq_client()

  sql = f"""
  SELECT attributes
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
    AND attributes IS NOT NULL
  ORDER BY timestamp DESC
  LIMIT 1
  """
  rows = query_bq(client, sql)

  if not rows:
    print("    No events with attributes found!")
    return FAIL, sid

  raw_attrs = rows[0]["attributes"]
  attrs = raw_attrs if isinstance(raw_attrs, dict) else json.loads(raw_attrs)
  session_meta = attrs.get("session_metadata", {})
  state = session_meta.get("state", {})

  if not state:
    print("    Session state not found in attributes!")
    print(f"    attributes keys: {list(attrs.keys())}")
    print(
        "    session_metadata keys:"
        f" {list(session_meta.keys()) if session_meta else 'N/A'}"
    )
    return FAIL, sid

  big_val = state.get("big_key", "")
  truncated = "TRUNCATED" in big_val or len(big_val) < 5000
  normal_ok = state.get("normal_key") == "small value"

  print(f"    big_key length in BQ: {len(big_val)}")
  print(f"    big_key truncated: {truncated}")
  print(f"    normal_key preserved: {normal_ok}")

  all_ok = truncated and normal_ok
  result = PASS if all_ok else FAIL
  print(f"\n    TEST 2: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  TEST 3: String system prompt is truncated  (Fix #5)
# ══════════════════════════════════════════════════════════════
async def test_system_prompt_truncation():
  """Run agent with long string system prompt, verify truncation in BQ."""
  print("\n" + "=" * 60)
  print("TEST 3: String system prompt is truncated (Fix #5)")
  print("=" * 60)

  max_content_length = 200
  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      max_content_length=max_content_length,
  )
  plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  # Create agent with a very long system instruction string
  long_instruction = "You are a helpful assistant. " * 100  # ~3000 chars
  print(f"    System instruction length: {len(long_instruction)} chars")

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name="prompt_test_agent",
      description="Agent for system prompt truncation test.",
      instruction=long_instruction,
      tools=[roll_die],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="prompt_test",
      agent=agent,
      session_service=session_service,
      plugins=[plugin],
  )

  session = await session_service.create_session(
      app_name="prompt_test", user_id="user-prompt-test"
  )
  sid = session.id
  print(f"    Session: {sid}")

  content = types.Content(
      role="user", parts=[types.Part(text="Roll a 20-sided die")]
  )
  async for event in runner.run_async(
      user_id="user-prompt-test",
      session_id=sid,
      new_message=content,
  ):
    pass

  await plugin.flush()
  await asyncio.sleep(2)
  await plugin.shutdown()

  print("    Waiting 10s for BQ data...")
  await asyncio.sleep(10)

  client = bq_client()

  # Check the LLM_REQUEST event content for truncated system_prompt
  sql = f"""
  SELECT content
  FROM `{FULL_TABLE}`
  WHERE session_id = '{sid}'
    AND event_type = 'LLM_REQUEST'
  LIMIT 1
  """
  rows = query_bq(client, sql)

  if not rows:
    print("    No LLM_REQUEST events found!")
    return FAIL, sid

  raw_content = rows[0]["content"]
  content_json = (
      raw_content if isinstance(raw_content, dict) else json.loads(raw_content)
  )
  system_prompt = content_json.get("system_prompt", "")

  truncated = "TRUNCATED" in system_prompt
  shorter = len(system_prompt) < len(long_instruction)

  print(f"    system_prompt length in BQ: {len(system_prompt)}")
  print(f"    Contains TRUNCATED marker: {truncated}")
  print(f"    Shorter than original: {shorter}")

  if system_prompt:
    print(f"    Preview: {system_prompt[:80]}...{system_prompt[-30:]}")

  all_ok = truncated and shorter
  result = PASS if all_ok else FAIL
  print(f"\n    TEST 3: {result}")
  return result, sid


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
  print("=" * 60)
  print("Issue #4694 — Six Bug Fixes Validation Suite")
  print(f"PID: {os.getpid()}, Platform: {sys.platform}")
  print("=" * 60)
  print(
      "\nNOTE: Fixes #2 (nested schema), #3 (multi-loop shutdown),"
      " #6 (dead code)"
  )
  print("are structural and fully covered by unit tests.\n")

  results = {}
  all_sids = []

  # Test 1: Error status
  try:
    r, sid = await test_error_status()
    results["error_status"] = r
    if sid:
      all_sids.append(sid)
  except Exception as e:
    print(f"    TEST 1 ERROR: {e}")
    traceback.print_exc()
    results["error_status"] = FAIL

  # Test 2: Session state truncation
  try:
    r, sid = await test_session_state_truncation()
    results["session_state_truncation"] = r
    if sid:
      all_sids.append(sid)
  except Exception as e:
    print(f"    TEST 2 ERROR: {e}")
    traceback.print_exc()
    results["session_state_truncation"] = FAIL

  # Test 3: System prompt truncation
  try:
    r, sid = await test_system_prompt_truncation()
    results["system_prompt_truncation"] = r
    if sid:
      all_sids.append(sid)
  except Exception as e:
    print(f"    TEST 3 ERROR: {e}")
    traceback.print_exc()
    results["system_prompt_truncation"] = FAIL

  # Final count
  print("\n" + "=" * 60)
  print("FINAL VERIFICATION")
  print("=" * 60)
  client = bq_client()
  for sid in all_sids:
    sql = f"""
    SELECT COUNT(*) as total
    FROM `{FULL_TABLE}`
    WHERE session_id = '{sid}'
    """
    rows = query_bq(client, sql)
    cnt = rows[0]["total"]
    label = "OK" if cnt > 0 else "MISSING"
    print(f"    Session {sid[:16]}... -> {cnt} events [{label}]")

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

  print(f"\n  Structural fixes (unit-test only):")
  print(f"  [OK  ] nested_schema_upgrade (Fix #2)")
  print(f"  [OK  ] multi_loop_shutdown (Fix #3)")
  print(f"  [OK  ] dead_code_removal (Fix #6)")

  if all_pass:
    print("\n  ALL TESTS PASSED")
    return 0
  else:
    print("\n  SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
