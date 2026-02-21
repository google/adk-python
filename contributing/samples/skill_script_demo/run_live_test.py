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

"""Live integration test: realistic skill workflows with Gemini.

Exercises the full skill lifecycle — discovery, loading, reference
reading, script inspection, and script execution — using both
Anthropic's mcp-builder skill and a lightweight python-helper skill.

Usage:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    python contributing/samples/skill_script_demo/run_live_test.py
"""

import asyncio
import os
import pathlib
import subprocess
import sys
import traceback

# Ensure the repo src is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "src"))

from google.adk import Agent
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types

_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

# Configure for Vertex AI
project = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project:
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    project = result.stdout.strip()
  except (FileNotFoundError, subprocess.TimeoutExpired):
    project = ""
  if project:
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")


def build_agent():
  """Build the agent with mcp-builder and python-helper skills."""
  mcp_builder = load_skill_from_dir(_SKILLS_DIR / "mcp-builder")
  python_helper = load_skill_from_dir(_SKILLS_DIR / "python-helper")

  toolset = SkillToolset(
      skills=[mcp_builder, python_helper],
      code_executor=UnsafeLocalCodeExecutor(),
  )

  return Agent(
      model="gemini-2.5-flash",
      name="skill_test_agent",
      description=(
          "An agent that uses skills for MCP development and Python utilities."
      ),
      tools=[toolset],
  )


# ── Event extraction helpers ────────────────────────────────────


def extract_tool_calls(events):
  """Extract tool call names from events."""
  names = []
  for ev in events:
    if not ev.content or not ev.content.parts:
      continue
    for part in ev.content.parts:
      if part.function_call:
        names.append(part.function_call.name)
  return names


def extract_tool_call_args(events):
  """Extract tool calls with their arguments."""
  calls = []
  for ev in events:
    if not ev.content or not ev.content.parts:
      continue
    for part in ev.content.parts:
      if part.function_call:
        calls.append({
            "name": part.function_call.name,
            "args": dict(part.function_call.args or {}),
        })
  return calls


def extract_tool_responses(events):
  """Extract function response payloads."""
  responses = []
  for ev in events:
    if not ev.content or not ev.content.parts:
      continue
    for part in ev.content.parts:
      if part.function_response:
        responses.append({
            "name": part.function_response.name,
            "response": part.function_response.response,
        })
  return responses


def extract_final_text(events):
  """Extract the final text response."""
  for ev in reversed(events):
    if not ev.content or not ev.content.parts:
      continue
    for part in ev.content.parts:
      if part.text:
        return part.text
  return ""


# ── Query runner ────────────────────────────────────────────────


async def run_query(runner, session_id, user_id, query):
  """Send a query and collect all events."""
  content = types.Content(
      role="user",
      parts=[types.Part.from_text(text=query)],
  )
  events = []
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
  ):
    events.append(event)
  return events


# ── Check engine ────────────────────────────────────────────────

SKILL_TOOLS = frozenset([
    "list_skills",
    "load_skill",
    "load_skill_resource",
    "execute_skill_script",
])


def check(tc, tool_calls, call_args, responses, text):
  """Run all checks for a test step. Returns (ok, messages)."""
  ok = True
  msgs = []

  # ── expect_tools: every listed tool must appear ──
  for t in tc.get("expect_tools", []):
    if t in tool_calls:
      msgs.append(f"  PASS: Used '{t}'")
    else:
      msgs.append(f"  FAIL: Expected '{t}' in {tool_calls}")
      ok = False

  # ── expect_any_tool: at least one must appear ──
  any_tools = tc.get("expect_any_tool", [])
  if any_tools:
    matched = [t for t in any_tools if t in tool_calls]
    if matched:
      msgs.append(f"  PASS: Used one of {any_tools}: {matched}")
    else:
      msgs.append(f"  FAIL: Expected one of {any_tools} in {tool_calls}")
      ok = False

  # ── expect_no_skill_tools ──
  if tc.get("expect_no_skill_tools"):
    used = [t for t in tool_calls if t in SKILL_TOOLS]
    if used:
      msgs.append(f"  FAIL: Should NOT have used: {used}")
      ok = False
    else:
      msgs.append("  PASS: No skill tools used")

  # ── expect_text_contains ──
  for s in tc.get("expect_text_contains", []):
    if s.lower() in text.lower():
      msgs.append(f"  PASS: Response contains '{s}'")
    else:
      msgs.append(f"  FAIL: Response missing '{s}'")
      ok = False

  # ── expect_text_any ──
  text_any = tc.get("expect_text_any", [])
  if text_any:
    found = [s for s in text_any if s.lower() in text.lower()]
    if found:
      msgs.append(f"  PASS: Response contains one of: {found}")
    else:
      msgs.append(f"  FAIL: Response missing all of: {text_any}")
      ok = False

  # ── expect_execute_script: verify execute call args ──
  ex = tc.get("expect_execute_script")
  if ex:
    exec_calls = [c for c in call_args if c["name"] == "execute_skill_script"]
    if not exec_calls:
      msgs.append(f"  FAIL: No execute_skill_script call in {tool_calls}")
      ok = False
    else:
      c = exec_calls[0]
      if "skill_name" in ex:
        actual = c["args"].get("skill_name")
        if actual == ex["skill_name"]:
          msgs.append(f"  PASS: skill_name='{actual}'")
        else:
          msgs.append(
              "  FAIL: skill_name: expected"
              f" '{ex['skill_name']}' got '{actual}'"
          )
          ok = False
      if "script_name" in ex:
        actual = c["args"].get("script_name", "")
        expected = ex["script_name"]
        if actual == expected or actual == f"scripts/{expected}":
          msgs.append(f"  PASS: script_name='{actual}'")
        else:
          msgs.append(
              f"  FAIL: script_name: expected '{expected}' got '{actual}'"
          )
          ok = False
      if "has_input_args" in ex:
        has = bool(c["args"].get("input_args"))
        if has == ex["has_input_args"]:
          msgs.append(f"  PASS: input_args present={has}")
        else:
          msgs.append(
              "  FAIL: input_args: expected"
              f" present={ex['has_input_args']} got {has}"
          )
          ok = False

  # ── expect_script_output: verify execution result ──
  out = tc.get("expect_script_output")
  if out:
    exec_resps = [r for r in responses if r["name"] == "execute_skill_script"]
    if not exec_resps:
      msgs.append("  FAIL: No execute_skill_script response")
      ok = False
    else:
      resp = exec_resps[0]["response"]
      expect_status = out.get("status", "success")
      actual_status = resp.get("status", "")
      if actual_status == expect_status:
        msgs.append(f"  PASS: Script status='{actual_status}'")
      else:
        msgs.append(
            "  FAIL: Script status: expected"
            f" '{expect_status}' got '{actual_status}'"
            f" error={resp.get('error', '')}"
        )
        ok = False

      stdout = resp.get("stdout", "")
      for s in out.get("stdout_contains", []):
        if s.lower() in stdout.lower():
          msgs.append(f"  PASS: stdout contains '{s}'")
        else:
          msgs.append(f"  FAIL: stdout missing '{s}'")
          ok = False

      stderr = resp.get("stderr", "")
      for s in out.get("stderr_contains", []):
        if s.lower() in stderr.lower():
          msgs.append(f"  PASS: stderr contains '{s}'")
        else:
          msgs.append(f"  FAIL: stderr missing '{s}'")
          ok = False

  # ── expect_resource_loaded: verify load_skill_resource resp ──
  res = tc.get("expect_resource_loaded")
  if res:
    res_resps = [r for r in responses if r["name"] == "load_skill_resource"]
    if not res_resps:
      msgs.append("  FAIL: No load_skill_resource response")
      ok = False
    else:
      for r in res_resps:
        content = r["response"].get("content", "")
        for s in res.get("content_contains", []):
          if s.lower() in content.lower():
            msgs.append(f"  PASS: Resource contains '{s}'")
          else:
            msgs.append(f"  FAIL: Resource missing '{s}'")
            ok = False

  return ok, msgs


# ── Test scenarios ──────────────────────────────────────────────


def get_test_scenarios():
  """Return test scenario groups."""
  return [
      # ─────────────────────────────────────────────────────────
      # Scenario 1: Discover both skills
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 1: Discover both skills",
          "shared_session": False,
          "steps": [{
              "name": "1a: List all available skills",
              "query": (
                  "What skills do you have? List them all"
                  " with their descriptions."
              ),
              "expect_any_tool": ["list_skills", "load_skill"],
              "expect_text_contains": [
                  "mcp-builder",
                  "python-helper",
              ],
          }],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 2: MCP development — load skill & read reference
      # ─────────────────────────────────────────────────────────
      {
          "name": (
              "Scenario 2: MCP dev workflow — load skill, read Python reference"
          ),
          "shared_session": True,
          "steps": [
              {
                  "name": (
                      "2a: Load mcp-builder and ask about"
                      " Python server patterns"
                  ),
                  "query": (
                      "I want to build an MCP server in"
                      " Python using FastMCP. Load the"
                      " mcp-builder skill and tell me the"
                      " four workflow phases."
                  ),
                  "expect_tools": ["load_skill"],
                  "expect_text_any": [
                      "Phase",
                      "Research",
                      "Implementation",
                      "Review",
                      "Evaluation",
                  ],
              },
              {
                  "name": "2b: Read the Python MCP server reference",
                  "query": (
                      "Now use load_skill_resource to read"
                      " references/python_mcp_server.md"
                      " from the mcp-builder skill."
                      " Summarize the key patterns it"
                      " recommends for implementing tools."
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_resource_loaded": {
                      "content_contains": [
                          "FastMCP",
                          "@mcp.tool",
                      ],
                  },
              },
          ],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 3: Inspect then execute connections.py
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 3: Inspect and execute mcp-builder connections.py",
          "shared_session": True,
          "steps": [
              {
                  "name": "3a: Inspect connections.py source",
                  "query": (
                      "Use load_skill_resource to show me"
                      " the source code of"
                      " scripts/connections.py from the"
                      " mcp-builder skill."
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_resource_loaded": {
                      "content_contains": [
                          "MCPConnection",
                          "create_connection",
                      ],
                  },
              },
              {
                  "name": "3b: Execute connections.py",
                  "query": (
                      "Please execute connections.py from"
                      " the mcp-builder skill using the"
                      " execute_skill_script tool. I want"
                      " to confirm the script loads without"
                      " errors."
                  ),
                  "expect_tools": ["execute_skill_script"],
                  "expect_execute_script": {
                      "skill_name": "mcp-builder",
                      "script_name": "connections.py",
                  },
                  "expect_script_output": {
                      "status": "success",
                  },
              },
          ],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 4: Read evaluation reference + example XML
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 4: Evaluation workflow — read eval guide & example",
          "shared_session": True,
          "steps": [
              {
                  "name": "4a: Read the evaluation guide reference",
                  "query": (
                      "Use load_skill_resource to read"
                      " references/evaluation.md from the"
                      " mcp-builder skill. Summarize the"
                      " key requirements for a good"
                      " evaluation question."
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_resource_loaded": {
                      "content_contains": ["qa_pair"],
                  },
                  "expect_text_any": [
                      "independent",
                      "read-only",
                      "verifiable",
                      "stable",
                      "non-destructive",
                  ],
              },
              {
                  "name": "4b: Read the example evaluation XML",
                  "query": (
                      "Now use load_skill_resource to read"
                      " scripts/example_evaluation.xml from"
                      " the mcp-builder skill and tell me"
                      " how many QA pairs it contains and"
                      " what topics they cover."
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_resource_loaded": {
                      "content_contains": [
                          "qa_pair",
                          "compound interest",
                      ],
                  },
                  "expect_text_any": ["5", "five"],
              },
          ],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 5: Execute fibonacci with python-helper
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 5: Execute python-helper fibonacci script",
          "shared_session": False,
          "steps": [{
              "name": "5a: Generate first 8 Fibonacci numbers",
              "query": (
                  "Use the python-helper skill to run"
                  " fibonacci.py and generate the first 8"
                  " Fibonacci numbers."
              ),
              "expect_tools": ["execute_skill_script"],
              "expect_execute_script": {
                  "skill_name": "python-helper",
                  "script_name": "fibonacci.py",
                  "has_input_args": True,
              },
              "expect_script_output": {
                  "status": "success",
                  "stdout_contains": ["Fibonacci"],
              },
              "expect_text_any": ["13", "0, 1, 1, 2, 3, 5, 8, 13"],
          }],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 6: Execute word_count with python-helper
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 6: Execute python-helper word_count script",
          "shared_session": False,
          "steps": [{
              "name": "6a: Analyze word frequency",
              "query": (
                  "Use the python-helper skill to run"
                  " word_count.py on the text: 'to be or"
                  " not to be that is the question'"
              ),
              "expect_tools": ["execute_skill_script"],
              "expect_execute_script": {
                  "skill_name": "python-helper",
                  "script_name": "word_count.py",
                  "has_input_args": True,
              },
              "expect_script_output": {
                  "status": "success",
                  "stdout_contains": [
                      "Total words",
                      "Unique words",
                  ],
              },
              "expect_text_any": ["to", "be"],
          }],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 7: Full multi-turn with MCP builder + execution
      # ─────────────────────────────────────────────────────────
      {
          "name": (
              "Scenario 7: Multi-turn — load mcp-builder,"
              " read reference, inspect script, execute"
          ),
          "shared_session": True,
          "steps": [
              {
                  "name": "7a: Load mcp-builder skill and read best practices",
                  "query": (
                      "Load the mcp-builder skill. Then"
                      " use load_skill_resource to read"
                      " references/mcp_best_practices.md."
                      " What does it say about tool naming?"
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_any_tool": [
                      "load_skill",
                      "load_skill_resource",
                  ],
                  "expect_resource_loaded": {
                      "content_contains": ["snake_case"],
                  },
                  "expect_text_any": [
                      "snake_case",
                      "prefix",
                      "naming",
                  ],
              },
              {
                  "name": "7b: Inspect connections.py source code",
                  "query": (
                      "Use load_skill_resource to read"
                      " scripts/connections.py from the"
                      " mcp-builder skill."
                  ),
                  "expect_tools": ["load_skill_resource"],
                  "expect_resource_loaded": {
                      "content_contains": [
                          "MCPConnection",
                      ],
                  },
              },
              {
                  "name": "7c: Execute connections.py we just inspected",
                  "query": (
                      "I know it only defines classes,"
                      " but please call"
                      " execute_skill_script with"
                      " skill_name='mcp-builder' and"
                      " script_name='connections.py'"
                      " anyway. If the mcp package is"
                      " missing the imports will raise"
                      " an ImportError, which is what"
                      " I want to check."
                  ),
                  "expect_tools": ["execute_skill_script"],
                  "expect_execute_script": {
                      "skill_name": "mcp-builder",
                      "script_name": "connections.py",
                  },
                  "expect_script_output": {
                      "status": "success",
                  },
              },
              {
                  "name": "7d: Now use python-helper to process some data",
                  "query": (
                      "Now switch to the python-helper"
                      " skill and use json_format.py to"
                      " pretty-print this JSON:"
                      ' {"tools":["list_tools",'
                      '"call_tool"],'
                      '"transport":"stdio"}'
                  ),
                  "expect_tools": ["execute_skill_script"],
                  "expect_execute_script": {
                      "skill_name": "python-helper",
                      "script_name": "json_format.py",
                      "has_input_args": True,
                  },
                  "expect_script_output": {
                      "status": "success",
                      "stdout_contains": [
                          "tools",
                          "transport",
                      ],
                  },
              },
          ],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 8: Agent routes to the right skill
      # ─────────────────────────────────────────────────────────
      {
          "name": (
              "Scenario 8: Agent picks mcp-builder for"
              " MCP questions, not python-helper"
          ),
          "shared_session": False,
          "steps": [{
              "name": "8a: MCP question routes to mcp-builder",
              "query": (
                  "I need to add pagination to my MCP"
                  " server's list endpoint. Load the"
                  " relevant skill and show me the"
                  " recommended pagination pattern."
              ),
              "expect_tools": ["load_skill"],
              "expect_text_any": [
                  "offset",
                  "limit",
                  "has_more",
                  "pagination",
                  "next_offset",
              ],
          }],
      },
      # ─────────────────────────────────────────────────────────
      # Scenario 9: Unrelated query does not use skills
      # ─────────────────────────────────────────────────────────
      {
          "name": "Scenario 9: Unrelated query avoids skills",
          "shared_session": False,
          "steps": [{
              "name": "9a: Simple math question",
              "query": "What is 17 * 23?",
              "expect_no_skill_tools": True,
              "expect_text_contains": ["391"],
          }],
      },
  ]


# ── Main runner ─────────────────────────────────────────────────


async def main():
  agent = build_agent()
  session_service = InMemorySessionService()
  runner = Runner(
      agent=agent,
      app_name="skill_live_test",
      session_service=session_service,
  )

  user_id = "test_user"
  scenarios = get_test_scenarios()
  total_passed = 0
  total_failed = 0
  failures = []

  for scenario in scenarios:
    print(f"\n{'#'*60}")
    print(f"  {scenario['name']}")
    print(f"{'#'*60}")

    shared_session = None
    if scenario["shared_session"]:
      shared_session = await session_service.create_session(
          app_name="skill_live_test", user_id=user_id
      )

    for step in scenario["steps"]:
      print(f"\n  {'─'*54}")
      print(f"  {step['name']}")
      print(f"  Query: {step['query'][:80]}...")

      session = shared_session or (
          await session_service.create_session(
              app_name="skill_live_test", user_id=user_id
          )
      )

      try:
        events = await run_query(runner, session.id, user_id, step["query"])
      except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        total_failed += 1
        failures.append((step["name"], str(e)))
        continue

      tool_calls = extract_tool_calls(events)
      call_args = extract_tool_call_args(events)
      responses = extract_tool_responses(events)
      text = extract_final_text(events)

      print(f"  Tool calls: {tool_calls}")
      for c in call_args:
        if c["name"] == "execute_skill_script":
          print(f"  Execute args: {c['args']}")
        elif c["name"] == "load_skill_resource":
          print(f"  Resource: {c['args'].get('path', '?')}")
      print(f"  Response: {text[:250]}...")

      ok, msgs = check(step, tool_calls, call_args, responses, text)
      for msg in msgs:
        print(msg)

      if ok:
        total_passed += 1
      else:
        total_failed += 1
        failures.append((step["name"], "assertion failure"))

  total = total_passed + total_failed
  print(f"\n{'='*60}")
  print(f"  RESULTS: {total_passed}/{total} passed")
  if failures:
    print(f"\n  Failures:")
    for name, reason in failures:
      print(f"    - {name}: {reason}")
  print(f"{'='*60}")
  return total_failed == 0


if __name__ == "__main__":
  success = asyncio.run(main())
  sys.exit(0 if success else 1)
