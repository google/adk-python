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

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import traceback

# Sentinel string used by verify_md.py to locate and split the coverage section
# out of run.py's stdout. Keep in sync with verify_md.py:COV_SECTION_HEADER.
COV_SECTION_HEADER = "📊 Phase 4: Code Coverage Report"

# Structured exit codes — consumed by verify_md.py to classify results without
# fragile string/emoji matching.  Keep in sync with verify_md.py:EXIT_* constants.
EXIT_SUCCESS = 0  # All phases passed
EXIT_LOAD_FAILURE = 1  # Failed to compile / load the snippet
EXIT_RUN_FAILURE = 2  # Loaded OK but the ADK component failed at runtime
EXIT_NO_COMPONENT = 3  # Loaded OK, no runnable ADK component found (load-only)

# --- Optional Coverage Integration ---
try:
  import coverage

  HAS_COVERAGE = True
except ImportError:
  HAS_COVERAGE = False

# --- Imports for ADK Inspection ---
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import Workflow
from google.genai import types


def load_target_module(file_path: Path):
  """Dynamically loads a Python file as a module, catching import/compilation/definition errors."""
  # Use the absolute path string as the key to avoid collisions when multiple
  # snippets share the same file stem or when the stem matches an installed package.
  module_name = str(file_path)
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not resolve module spec for file '{file_path.name}'"
    )

  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module

  # Executing the module runs all top-level code, which will catch:
  # - SyntaxError / IndentationError
  # - ImportError (e.g. from google.adk.workflow import build_node)
  # - ValidationError (e.g. instantiating Workflow with invalid edges)
  try:
    spec.loader.exec_module(module)
  except Exception:
    # Remove the partially-initialised module so a broken entry is never
    # left in sys.modules for the lifetime of this process.  This matters
    # when run.py is imported in-process (e.g. from a test harness) rather
    # than invoked as a subprocess.
    sys.modules.pop(module_name, None)
    raise
  return module


def discover_adk_component(module):
  """Scans the module namespace to discover runnable ADK components, prioritizing root components.

  Uses two passes to correctly identify root agents regardless of the order
  in which names appear in ``vars(module)``:

  * Pass 1 — collect every Workflow, Agent, and App in the module namespace.
  * Pass 2 — build the full set of sub-agent IDs from *all* collected agents,
    then filter to find agents that are not sub-agents of any other agent.

  Without the two-pass approach, a root agent whose variable name is seen
  before its sub-agents (e.g. ``root`` defined above ``child`` in the file)
  would be encountered first, before ``child``'s own sub-agents are registered,
  causing incorrect root detection.
  """
  workflows = []
  agents = []
  apps = []

  # Pass 1: collect all candidate components.
  #
  # Use vars(module) rather than inspect.getmembers(module) because
  # getmembers() invokes every attribute getter and silently swallows any
  # Exception raised by broken descriptors or properties — a snippet that
  # defines an Agent behind a faulty @property would simply be missing from
  # the scan with no error or log entry.  vars(module) reads the module's
  # __dict__ directly, which never triggers descriptors and never suppresses
  # exceptions, giving us an accurate view of module-level names.
  for obj in vars(module).values():
    if isinstance(obj, Workflow):
      workflows.append(obj)
    elif isinstance(obj, BaseAgent):
      agents.append(obj)
    elif isinstance(obj, App):
      apps.append(obj)

  # 1. Prefer Workflow
  if workflows:
    return workflows[0], "Workflow"

  # Pass 2: build the complete sub-agent ID set now that all agents are known,
  # then select the root (any agent not listed as a sub-agent of another).
  #
  # Read sub_agents into a local snapshot rather than calling the attribute
  # twice.  Calling it twice is unsafe when sub_agents is a non-idempotent
  # property: the first call (guard) and the second call (iteration) could
  # return different objects, causing id() values to diverge and root
  # detection to silently misfire.
  sub_agent_ids: set[int] = set()
  for agent in agents:
    children = getattr(agent, "sub_agents", None) or []
    for sub in children:
      sub_agent_ids.add(id(sub))

  # 2. Find root Agent (not a sub-agent of any other agent in the module)
  root_agents = [a for a in agents if id(a) not in sub_agent_ids]
  if root_agents:
    return root_agents[0], "Agent"

  # 3. Fall back to App
  if apps:
    return apps[0], "App"

  return None, None


async def run_component(component, component_type, test_input):
  """Unified runner to execute the discovered component."""
  print(f"\n🔍 Discovered ADK {component_type} in target file.")
  print(f"🚀 Running execution test with input: '{test_input}'...\n")

  if component_type == "App":
    runnable_node = getattr(component, "root_agent", None)
    if runnable_node is None:
      raise AttributeError(
          f"App instance has no 'root_agent' attribute. "
          f"Ensure the App is constructed with a root_agent argument."
      )
  else:
    runnable_node = component

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="runnability_test",
      node=runnable_node,
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name="runnability_test", user_id="tester"
  )

  user_message = types.Content(
      parts=[types.Part(text=str(test_input))], role="user"
  )

  async for event in runner.run_async(
      user_id="tester", session_id=session.id, new_message=user_message
  ):
    print(f"🎬 [Event] Author: {event.author}")
    if event.output:
      print(f"🔹 Output: {event.output}")
    if hasattr(event, "content") and event.content and event.content.parts:
      text = "".join(p.text for p in event.content.parts if p.text)
      if text:
        print(f"📝 Content Output:\n{'-'*40}\n{text}\n{'-'*40}")


def main():
  parser = argparse.ArgumentParser(
      description="Generalized ADK Runnability & Loadability Tester"
  )
  parser.add_argument(
      "file",
      type=str,
      help="Path to the python file containing the agent/workflow to test",
  )
  args = parser.parse_args()

  file_path = Path(args.file).resolve()
  if not file_path.exists():
    print(f"❌ Error: File '{file_path}' does not exist.")
    sys.exit(EXIT_LOAD_FAILURE)

  print(f"🔬 Testing file: {file_path.name}")
  print("=" * 60)

  # Initialize coverage programmatically to track ONLY the target file.
  #
  # Implementation note: snippets are loaded via importlib/exec_module, which
  # CPython's sys.settrace-based tracer instruments correctly *only* if the
  # tracer is active before the module's code object is compiled and executed.
  # Starting coverage here — before load_target_module() — satisfies that
  # requirement. The `include` filter ensures no ADK library code is counted.
  cov = None
  if HAS_COVERAGE:
    cov = coverage.Coverage(
        branch=True,
        data_file=None,  # Keep coverage data in-memory only, no .coverage file needed
        include=[str(file_path)],  # Scope collection to the snippet file only
    )
    cov.start()
  else:
    print(
        "ℹ️  Install 'coverage' package to enable automated code coverage"
        " reporting."
    )

  # exit_code is set by each phase and consumed inside the finally block so
  # that coverage reporting always runs before the process exits.  Using a
  # mutable list as a simple cell lets the finally clause read the value set
  # by any code path (normal completion, early break-out via a flag, or an
  # unexpected exception) without requiring nonlocal or a class wrapper.
  exit_code = [EXIT_SUCCESS]

  try:
    # 1. Test Loadability (Imports, Syntax, Instantiation/Validation)
    print("📋 Phase 1: Loading & Compiling...")
    try:
      module = load_target_module(file_path)
      print(
          f"✅ Load Success: '{file_path.name}' compiled and loaded without any"
          " issues."
      )
    except Exception:
      print(f"❌ Load Failure: Failed to compile/load '{file_path.name}':")
      print("-" * 60)
      traceback.print_exc(file=sys.stdout)
      print("-" * 60)
      exit_code[0] = EXIT_LOAD_FAILURE
      # Do NOT return here.  Fall through to the finally block so that
      # coverage is reported and sys.exit() is called with the correct code.
      # The module variable is not set, so we skip phases 2–3 via the flag.
    else:
      # 2. Discover Component (only reached when load succeeded)
      print("\n📋 Phase 2: Component Discovery...")
      component, comp_type = discover_adk_component(module)
      if not component:
        print(
            "➖ NO ADK COMPONENT: No module-level Workflow, Agent, or App"
            f" instance found in '{file_path.name}'."
        )
        print(
            "   Runnability test skipped. To enable it, assign a Workflow,"
            " Agent, or App"
        )
        print(
            "   to a module-level variable (e.g. `agent = Agent(...)`). The"
            " variable name"
        )
        print(
            "   does not matter — the runner detects it automatically via"
            " vars(module)."
        )
        print(
            "ℹ️  Coverage below reflects load-time execution only (module-level"
            " statements)."
        )
        exit_code[0] = EXIT_NO_COMPONENT
      else:
        # Get test input from module, or fallback
        test_input = getattr(module, "test_input", "Test input topic")

        # 3. Test Runnability
        print(f"\n📋 Phase 3: Executing {comp_type}...")
        try:
          # asyncio.run() creates a fresh event loop each time, so it will raise
          # RuntimeError if a loop is already running (e.g. the snippet called
          # asyncio.run() at module level without an __main__ guard).
          # We catch that specific case and report it clearly, rather than using
          # the deprecated asyncio.get_event_loop() API (removed in Python 3.12+).
          asyncio.run(run_component(component, comp_type, test_input))
          print(
              f"\n✅ Run Success: Component '{comp_type}' executed"
              " successfully."
          )
        except RuntimeError as e:
          if "event loop" in str(e).lower():
            print(
                f"\n❌ Run Failure: An event loop conflict was detected after"
                f" module load."
            )
            print(
                "   The snippet likely called asyncio.run() at module level,"
                " which"
            )
            print(
                "   conflicts with the runner's own event loop. Wrap top-level"
                " async"
            )
            print(
                "   calls in an `if __name__ == '__main__':` guard, or"
                " annotate the"
            )
            print("   snippet with <!-- verify-snippets: ignore -->.")
          else:
            print(f"\n❌ Run Failure: Component failed during execution:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
          exit_code[0] = EXIT_RUN_FAILURE
        except Exception:
          print(f"\n❌ Run Failure: Component failed during execution:")
          print("-" * 60)
          traceback.print_exc(file=sys.stdout)
          print("-" * 60)
          exit_code[0] = EXIT_RUN_FAILURE

  finally:
    # Coverage reporting runs here so it is guaranteed to execute on every
    # code path: normal completion, load failure, no-component, run failure.
    if cov:
      cov.stop()
      print(f"\n{COV_SECTION_HEADER} (Target File)")
      print("=" * 60)
      try:
        # Report coverage of the target file directly to stdout
        cov.report(morfs=[str(file_path)], file=sys.stdout)
      except coverage.exceptions.NoDataError:
        print(
            "⚠️  No coverage data collected (compilation or execution failed"
            " early)."
        )
      except Exception as ce:
        print(f"⚠️  Failed to generate coverage report: {ce}")
      print("=" * 60)
    # Only call sys.exit for non-zero codes.  If exit_code is EXIT_SUCCESS
    # we return normally so that any exception currently propagating out of
    # the try block is not silently replaced by a SystemExit raised here.
    if exit_code[0] != EXIT_SUCCESS:
      sys.exit(exit_code[0])


if __name__ == "__main__":
  main()
