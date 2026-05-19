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

import json
import os
from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from unittest import mock

from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events.event import Event as AdkEvent
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import alias_generators
import pytest

EXCLUDED_EVENT_FIELDS = {
    "id",
    "timestamp",
    "invocation_id",
    "model_version",
    "finish_reason",
    "usage_metadata",
    "avg_logprobs",
    "cache_metadata",
    "logprobs_result",
    "citation_metadata",
}


# Read target folder from environment
def get_test_files(
    target_folder: str | None = None,
) -> list[pytest.ParameterSet]:
  """Returns list of (agent_dir, test_file_path) recursively."""
  folder = target_folder or os.environ.get("ADK_TEST_FOLDER")
  if not folder:
    return []
  target_dir = Path(folder)
  if not target_dir.exists():
    return []

  samples_dir = (
      Path(__file__).parent.parent.parent.parent.parent
      / "contributing"
      / "samples"
  )

  results = []
  for test_file in target_dir.rglob("tests/*.json"):
    agent_dir = test_file.parent.parent
    # Verify it looks like an agent directory
    if (
        (agent_dir / "agent.py").exists()
        or (agent_dir / "__init__.py").exists()
        or (agent_dir / "root_agent.yaml").exists()
    ):
      try:
        rel_dir = agent_dir.relative_to(samples_dir)
        test_id = f"{rel_dir}/{test_file.name}"
      except ValueError:
        test_id = f"{agent_dir.name}/{test_file.name}"

      if test_file.stem.endswith("_xfail"):
        results.append(
            pytest.param(
                agent_dir, test_file, id=test_id, marks=pytest.mark.xfail
            )
        )
      else:
        results.append(pytest.param(agent_dir, test_file, id=test_id))
  return results


class MockModel(BaseLlm):
  model: str = "mock"
  requests: list[LlmRequest] = []
  responses: list[LlmResponse] = []
  response_index: int = -1

  @classmethod
  def create(cls, contents: list[types.Content]):
    llm_responses = [LlmResponse(content=content) for content in contents]
    return cls(responses=llm_responses)

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["mock"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.response_index += 1
    self.requests.append(llm_request)
    yield self.responses[self.response_index]


class InMemoryRunner:

  def __init__(self, root_agent=None, app=None):
    if app:
      self.app_name = app.name
      self.runner = Runner(
          app=app,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
      )
    else:
      self.app_name = "test_app"
      self.runner = Runner(
          app_name="test_app",
          agent=root_agent,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
      )
    self.session_id = None

  @property
  def session(self):
    if not self.session_id:
      session = self.runner.session_service.create_session_sync(
          app_name=self.app_name, user_id="test_user"
      )
      self.session_id = session.id
      return session
    return self.runner.session_service.get_session_sync(
        app_name=self.app_name, user_id="test_user", session_id=self.session_id
    )

  def run(self, new_message) -> list[AdkEvent]:
    content = (
        new_message
        if isinstance(new_message, types.Content)
        else types.Content(
            role="user", parts=[types.Part.from_text(text=new_message)]
        )
    )
    return list(
        self.runner.run(
            user_id=self.session.user_id,
            session_id=self.session.id,
            new_message=content,
        )
    )


def normalize_events(events, is_json=False):
  normalized = []
  for e in events:
    if is_json:
      d = dict(e)
      for k in EXCLUDED_EVENT_FIELDS:
        d.pop(k, None)
        d.pop(alias_generators.to_camel(k), None)
      d = {k: v for k, v in d.items() if v is not None}
    else:
      d = e.model_dump(
          mode="json",
          by_alias=True,
          exclude=EXCLUDED_EVENT_FIELDS,
          exclude_none=True,
      )

    if "content" in d and isinstance(d["content"], dict):
      content = d["content"]
      if "parts" in content and isinstance(content["parts"], list):
        for part in content["parts"]:
          if isinstance(part, dict) and "thoughtSignature" in part:
            del part["thoughtSignature"]

    if "longRunningToolIds" in d:
      if isinstance(d["longRunningToolIds"], list):
        if not d["longRunningToolIds"]:
          del d["longRunningToolIds"]
        else:
          d["longRunningToolIds"] = sorted(d["longRunningToolIds"])

    if "actions" in d:
      actions = d["actions"]
      if isinstance(actions, dict):
        # Remove empty dicts inside actions
        for k in list(actions.keys()):
          if actions[k] == {}:
            del actions[k]
        # If actions itself is now empty, remove it!
        if not actions:
          del d["actions"]

    actions = d.get("actions", {})
    state_delta = actions.get("stateDelta", {}) if actions else {}
    if state_delta:
      keys_to_remove = [k for k in state_delta if k.endswith("_join_state")]
      for k in keys_to_remove:
        del state_delta[k]

    normalized.append(d)
  return normalized


def make_sort_key(d):
  node_path = d.get("nodeInfo", {}).get("path", "")
  author = d.get("author", "")
  return (author, node_path, json.dumps(d, sort_keys=True))


def _make_nodes_sequential(obj, visited=None):
  if visited is None:
    visited = set()

  if id(obj) in visited:
    return
  visited.add(id(obj))

  from google.adk.workflow._parallel_worker import _ParallelWorker
  from google.adk.workflow._workflow import Workflow

  if isinstance(obj, Workflow):
    obj.max_concurrency = 1
    if obj.graph and obj.graph.nodes:
      for node in obj.graph.nodes:
        _make_nodes_sequential(node, visited)
  elif isinstance(obj, _ParallelWorker):
    obj.max_concurrency = 1
    if hasattr(obj, "_node"):
      _make_nodes_sequential(obj._node, visited)


def _extract_user_content(event: dict) -> Optional[types.Content]:
  """Extracts user content from an event dict and returns a types.Content object.

  Agent-emitted user-role events (e.g., task FRs synthesized by the Task
  Delegation API wrapper) are skipped — those are produced by the agent
  itself and carry a non-empty ``nodeInfo.path``.  Re-feeding them to
  ``runner.run`` would trigger extra LLM calls.
  """
  if event.get("author") != "user":
    return None

  # Real external user input has no node path; agent-emitted events do.
  if event.get("nodeInfo", {}).get("path"):
    return None

  content_dict = event.get("content", {})
  if not content_dict:
    return None

  parts = content_dict.get("parts", [])
  real_parts = []
  for p in parts:
    if "functionResponse" in p:
      fr = p["functionResponse"]
      real_parts.append(
          types.Part(
              function_response=types.FunctionResponse(
                  id=fr.get("id"),
                  name=fr.get("name"),
                  response=fr.get("response"),
              )
          )
      )
    elif "text" in p:
      real_parts.append(types.Part(text=p["text"]))
    elif "functionCall" in p:
      fc = p["functionCall"]
      real_parts.append(
          types.Part(
              function_call=types.FunctionCall(
                  id=fc.get("id"),
                  name=fc.get("name"),
                  args=fc.get("args"),
              )
          )
      )

  if real_parts:
    return types.Content(role="user", parts=real_parts)
  return None


def _remap_node_path(path: str, id_map: dict[str, str]) -> str:
  """Rewrite ``<name>@<id>`` segments in a node path using ``id_map``.

  Path segments encode ``<node_name>@<run_id>``; when the run_id was an
  LLM-generated FC id, it gets canonicalized to ``fc-N`` via ``id_map``.
  Segments without ``@`` and ids not in ``id_map`` pass through unchanged.
  """
  segments = []
  for seg in path.split("/"):
    if "@" in seg:
      name, rid = seg.split("@", 1)
      if rid in id_map:
        seg = f"{name}@{id_map[rid]}"
    segments.append(seg)
  return "/".join(segments)


def _normalize_ids(events: list[AdkEvent]) -> list[AdkEvent]:
  """Filters partial events and normalizes event, function call, and response IDs."""
  events = [e for e in events if not getattr(e, "partial", False)]

  # Re-assign sequential event IDs
  for i, e in enumerate(events, 1):
    e.id = f"e-{i}"

  # Post-process all events to inject deterministic function IDs
  final_fc_counter = 0
  final_orig_to_new_id = {}
  for e in events:
    for fc in e.get_function_calls():
      orig_id = fc.id
      final_fc_counter += 1
      new_id = f"fc-{final_fc_counter}"
      final_orig_to_new_id[orig_id] = new_id
      fc.id = new_id
      if e.long_running_tool_ids:
        e.long_running_tool_ids = {
            new_id if tid == orig_id else tid for tid in e.long_running_tool_ids
        }
      if fc.args:
        for k, v in fc.args.items():
          if v == orig_id:
            fc.args[k] = new_id

  # Pass 2: Update actions and user responses in all events
  call_name_to_ids = {}
  for e in events:
    for fc in e.get_function_calls():
      call_name_to_ids.setdefault(fc.name, []).append(fc.id)

    if getattr(e, "branch", None) and e.branch.startswith("task:"):
      parts = e.branch.split(":")
      if len(parts) > 1:
        fc_id = parts[1]
        if fc_id in final_orig_to_new_id:
          e.branch = f"task:{final_orig_to_new_id[fc_id]}"

    # Task wrappers stamp isolation_scope with the dispatching FC's
    # id (random at run time) and ``node_info.path`` encodes
    # ``<name>@<fc.id>`` for the same id — remap both.
    if e.isolation_scope in final_orig_to_new_id:
      e.isolation_scope = final_orig_to_new_id[e.isolation_scope]
    if e.node_info.path:
      e.node_info.path = _remap_node_path(
          e.node_info.path, final_orig_to_new_id
      )
    if e.node_info.output_for:
      e.node_info.output_for = [
          _remap_node_path(pth, final_orig_to_new_id)
          for pth in e.node_info.output_for
      ]

    if e.content and e.content.parts:
      for part in e.content.parts:
        if part.function_response:
          name = part.function_response.name
          if name in call_name_to_ids and call_name_to_ids[name]:
            part.function_response.id = call_name_to_ids[name].pop(0)
          elif part.function_response.id in final_orig_to_new_id:
            part.function_response.id = final_orig_to_new_id[
                part.function_response.id
            ]
        # Tool-confirmation FCs nest the original FC's id inside their
        # args; remap so the confirmation event aligns with the
        # canonical fc-N id of the call it confirms.
        if part.function_call and part.function_call.args:
          _remap_ids_in_args(part.function_call.args, final_orig_to_new_id)

    # actions.requested_tool_confirmations is keyed by the FC id of the
    # tool call awaiting confirmation; remap the keys to canonical ids.
    if e.actions and e.actions.requested_tool_confirmations:
      e.actions.requested_tool_confirmations = {
          final_orig_to_new_id.get(k, k): v
          for k, v in e.actions.requested_tool_confirmations.items()
      }

  return events


def _remap_ids_in_args(value: Any, id_map: dict[str, str]) -> None:
  """Walk a FC ``args`` value and remap any ``id`` field that names an FC.

  Tool-confirmation FCs (``adk_request_confirmation``) carry the
  original FC as ``args.originalFunctionCall``; its ``id`` needs to be
  remapped to the canonical ``fc-N`` value just like the top-level FC id.
  """
  if isinstance(value, dict):
    for k, v in list(value.items()):
      if k == "id" and isinstance(v, str) and v in id_map:
        value[k] = id_map[v]
      else:
        _remap_ids_in_args(v, id_map)
  elif isinstance(value, list):
    for item in value:
      _remap_ids_in_args(item, id_map)


@pytest.mark.parametrize(
    "agent_dir, test_file",
    get_test_files(),
)
def test_agent_replay(agent_dir, test_file, monkeypatch):
  # Add agent_dir.parent to sys.path so relative imports work
  import sys

  sys_path_saved = list(sys.path)
  sys.path.insert(0, str(agent_dir.parent))

  try:
    import random

    random.seed(42)

    loader = AgentLoader(str(agent_dir.parent))
    loader.remove_agent_from_cache(agent_dir.name)
    agent_or_app = loader.load_agent(agent_dir.name)

    root_agent = (
        agent_or_app.root_agent
        if isinstance(agent_or_app, App)
        else agent_or_app
    )
    _make_nodes_sequential(root_agent)

    with open(test_file, "r") as f:
      session_data = json.load(f)

    events_data = session_data.get("events", [])
    if not events_data:
      pytest.skip(f"No events in {test_file}")

    first_event = events_data[0]
    user_message = ""
    if first_event.get("author") == "user":
      parts = first_event.get("content", {}).get("parts", [])
      if parts and "text" in parts[0]:
        user_message = parts[0]["text"]

    if not user_message:
      pytest.skip(f"Could not find user message in {test_file}")

    expected_events = events_data[1:]

    import re

    parallel_pattern = re.compile(r"^(.+)__(\d+)$")

    all_responses = []
    last_was_set_model_response = False
    for ev in expected_events:
      if "content" in ev:
        content_dict = ev["content"]
        role = content_dict.get("role")

        if role == "user":
          parts = content_dict.get("parts", [])
          for part in parts:
            if "functionResponse" in part:
              func_resp = part["functionResponse"]
              if func_resp.get("name") == "set_model_response":
                last_was_set_model_response = True

        elif role == "model":
          if last_was_set_model_response:
            last_was_set_model_response = False
            continue

          try:
            content_obj = types.Content.model_validate(content_dict)
            all_responses.append(
                {"author": ev.get("author", ""), "content": content_obj}
            )
          except Exception:
            pass

    mock_responses = []
    current_parallel_base = None
    current_parallel_items = []

    for resp in all_responses:
      match = parallel_pattern.match(resp["author"])
      if match:
        base_name, index = match.groups()
        index = int(index)

        if current_parallel_base and current_parallel_base != base_name:
          # Flush previous parallel group
          current_parallel_items.sort(key=lambda x: x[0])
          mock_responses.extend([x[1] for x in current_parallel_items])
          current_parallel_items = []

        current_parallel_base = base_name
        current_parallel_items.append((index, resp["content"]))
      else:
        if current_parallel_base:
          # Flush previous parallel group
          current_parallel_items.sort(key=lambda x: x[0])
          mock_responses.extend([x[1] for x in current_parallel_items])
          current_parallel_items = []
          current_parallel_base = None

        mock_responses.append(resp["content"])

    # Flush last group
    if current_parallel_base:
      current_parallel_items.sort(key=lambda x: x[0])
      mock_responses.extend([x[1] for x in current_parallel_items])

    if mock_responses:
      mock_model = MockModel.create(contents=mock_responses)

      async def mock_gen_async(instance, llm_request, stream=False):
        async for resp in mock_model.generate_content_async(
            llm_request, stream
        ):
          yield resp

      from google.adk.models.base_llm import BaseLlm
      from google.adk.models.google_llm import Gemini

      monkeypatch.setattr(BaseLlm, "generate_content_async", mock_gen_async)
      monkeypatch.setattr(Gemini, "generate_content_async", mock_gen_async)

    # Make RequestInput IDs deterministic during replay as well
    fc_counter = 0

    def get_next_fc_id():
      nonlocal fc_counter
      fc_counter += 1
      return f"fc-{fc_counter}"

    runner = (
        InMemoryRunner(app=agent_or_app)
        if isinstance(agent_or_app, App)
        else InMemoryRunner(root_agent=agent_or_app)
    )

    # Extract all function call IDs from old events
    old_fc_ids = []
    for ev in events_data:
      content = ev.get("content", {})
      parts = content.get("parts", []) if isinstance(content, dict) else []
      for p in parts:
        if isinstance(p, dict) and "functionCall" in p:
          fc = p["functionCall"]
          if isinstance(fc, dict) and "id" in fc:
            old_fc_ids.append(fc["id"])

    orig_to_new_id = {}
    fc_counter = 0
    mapping_counter = 0

    actual_events = []
    import random

    mocks_data = session_data.get("mocks", {})
    if mocks_data:
      if "random.random" in mocks_data:
        random_values = list(mocks_data["random.random"])

        def mock_random():
          if random_values:
            return random_values.pop(0)
          return 0.8

        monkeypatch.setattr(random, "random", mock_random)

      if "random.randint" in mocks_data:
        randint_values = list(mocks_data["random.randint"])

        def mock_randint(a, b):
          if randint_values:
            return randint_values.pop(0)
          return b

        monkeypatch.setattr(random, "randint", mock_randint)
    else:
      random.seed(42)
    first_run_events = runner.run(user_message)

    # Post-process events to inject deterministic function IDs
    for e in first_run_events:
      for fc in e.get_function_calls():
        # Build mapping from old IDs to new agent IDs
        if mapping_counter < len(old_fc_ids):
          old_id = old_fc_ids[mapping_counter]
          orig_to_new_id[old_id] = fc.id
          mapping_counter += 1

    actual_events.extend(first_run_events)

    for event in events_data[1:]:
      if event.get("author") == "user":
        content = _extract_user_content(event)
        if content:
          # Update function response IDs if mapped
          if content.parts:
            for part in content.parts:
              if (
                  part.function_response
                  and part.function_response.id in orig_to_new_id
              ):
                part.function_response.id = orig_to_new_id[
                    part.function_response.id
                ]

          actual_events.append(
              AdkEvent(
                  author="user",
                  content=content,
              )
          )
          next_run_events = runner.run(content)

          # Post-process events to inject deterministic function IDs
          for e in next_run_events:
            for fc in e.get_function_calls():
              # Build mapping from old IDs to new agent IDs
              if mapping_counter < len(old_fc_ids):
                old_id = old_fc_ids[mapping_counter]
                orig_to_new_id[old_id] = fc.id
                mapping_counter += 1

          actual_events.extend(next_run_events)

    actual_events = _normalize_ids(actual_events)

    actual_dicts = normalize_events(actual_events, is_json=False)
    expected_dicts = normalize_events(expected_events, is_json=True)

    actual_dicts.sort(key=make_sort_key)
    expected_dicts.sort(key=make_sort_key)

    assert actual_dicts == expected_dicts
  finally:
    sys.path = sys_path_saved


def rebuild_tests(path: str):
  """Discovers test files and rebuilds them by running the agent live."""
  import json
  import sys
  import time

  from google.adk.apps.app import App
  from google.adk.events.event import Event as AdkEvent
  from google.genai import types

  path_obj = Path(path)
  if path_obj.is_dir():
    folder = path
    expected_name = None
  else:
    folder = str(path_obj.parent.parent)
    expected_name = path_obj.name

  test_files = get_test_files(folder)
  if not test_files:
    print(f"No test files found in {folder}")
    return

  for item in test_files:
    agent_dir, test_file = item.values
    if expected_name and test_file.name != expected_name:
      continue
    print(f"Rebuilding {test_file}...")

    # Add agent_dir.parent to sys.path so relative imports work
    sys_path_saved = list(sys.path)
    sys.path.insert(0, str(agent_dir.parent))

    try:
      import random

      loader = AgentLoader(str(agent_dir.parent))
      loader.remove_agent_from_cache(agent_dir.name)
      agent_or_app = loader.load_agent(agent_dir.name)

      root_agent = (
          agent_or_app.root_agent
          if isinstance(agent_or_app, App)
          else agent_or_app
      )
      _make_nodes_sequential(root_agent)

      with open(test_file, "r") as f:
        session_data = json.load(f)

      events_data = session_data.get("events", [])
      if not events_data:
        print(f"No events in {test_file}, skipping.")
        continue

      # Extract user messages
      user_messages = []
      for event in events_data:
        content = _extract_user_content(event)
        if content:
          user_messages.append(content)

      if not user_messages:
        print(f"No user messages found in {test_file}, skipping.")
        continue

      runner = (
          InMemoryRunner(app=agent_or_app)
          if isinstance(agent_or_app, App)
          else InMemoryRunner(root_agent=agent_or_app)
      )

      new_events = []
      inv_counter = 1

      def mock_inv_id():
        nonlocal inv_counter
        res = f"i-{inv_counter}"
        inv_counter += 1
        return res

      ev_counter = 1

      def mock_ev_id():
        nonlocal ev_counter
        res = f"e-{ev_counter}"
        ev_counter += 1
        return res

      fc_counter = 0
      orig_to_new_id = {}

      # Extract all function call IDs and response IDs from old events
      old_fc_ids = []
      old_fr_ids = []
      for ev in events_data:
        content = ev.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for p in parts:
          if isinstance(p, dict):
            if "functionCall" in p:
              fc = p["functionCall"]
              if isinstance(fc, dict) and "id" in fc:
                old_fc_ids.append(fc["id"])
            elif "functionResponse" in p:
              fr = p["functionResponse"]
              if isinstance(fr, dict) and "id" in fr:
                old_fr_ids.append(fr["id"])

      def get_next_fc_id():
        nonlocal fc_counter
        fc_counter += 1
        new_id = f"fc-{fc_counter}"
        if fc_counter <= len(old_fc_ids):
          orig_id = old_fc_ids[fc_counter - 1]
          orig_to_new_id[orig_id] = new_id
        if fc_counter <= len(old_fr_ids):
          orig_fr_id = old_fr_ids[fc_counter - 1]
          orig_to_new_id[orig_fr_id] = new_id
        return new_id

      with (
          mock.patch(
              "google.adk.runners.new_invocation_context_id",
              side_effect=mock_inv_id,
          ),
          mock.patch(
              "google.adk.events.event.Event.new_id", side_effect=mock_ev_id
          ),
          mock.patch(
              "google.adk.flows.llm_flows.functions.generate_client_function_call_id",
              side_effect=get_next_fc_id,
          ),
          mock.patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "rebuild"}),
      ):
        random.seed(42)
        for msg in user_messages:

          # Update function response IDs if mapped
          if msg.parts:
            for part in msg.parts:
              if (
                  part.function_response
                  and part.function_response.id in orig_to_new_id
              ):
                part.function_response.id = orig_to_new_id[
                    part.function_response.id
                ]

          # Create user event
          user_ev = AdkEvent(
              author="user",
              content=msg,
          )

          run_events = runner.run(msg)

          # Build mapping from old IDs to new agent IDs
          for e in run_events:
            for fc in e.get_function_calls():
              if fc_counter < len(old_fc_ids):
                old_id = old_fc_ids[fc_counter]
                orig_to_new_id[old_id] = fc.id
                fc_counter += 1

          # Set invocation_id from runner's output if available
          if run_events:
            user_ev.invocation_id = run_events[0].invocation_id

          new_events.append(user_ev)
          new_events.extend(run_events)

      new_events = _normalize_ids(new_events)

      # Convert to dicts
      # Also exclude timestamp to make it deterministic
      new_events_dicts = [
          e.model_dump(
              mode="json",
              by_alias=True,
              exclude_none=True,
              exclude={
                  "timestamp",
                  "usage_metadata",
                  "model_version",
                  "avg_logprobs",
                  "cache_metadata",
                  "logprobs_result",
                  "citation_metadata",
              },
          )
          for e in new_events
      ]

      # Clean up thoughtSignature if present
      for ev in new_events_dicts:
        if "content" in ev and isinstance(ev["content"], dict):
          content = ev["content"]
          if "parts" in content and isinstance(content["parts"], list):
            for part in content["parts"]:
              if isinstance(part, dict) and "thoughtSignature" in part:
                del part["thoughtSignature"]

      # Clean up empty actions items and actions itself if empty
      for ev in new_events_dicts:
        if "actions" in ev:
          actions = ev["actions"]
          ev["actions"] = {
              k: v
              for k, v in actions.items()
              if not (isinstance(v, dict) and not v)
          }
          if not ev["actions"]:
            del ev["actions"]

      # Update session data
      session_data["events"] = new_events_dicts
      session_data.pop("lastUpdateTime", None)

      # Write back to file
      with open(test_file, "w") as f:
        json.dump(session_data, f, indent=2, sort_keys=True)

      print(f"Successfully rebuilt {test_file}")

    except Exception as e:
      print(f"Error rebuilding {test_file}: {e}")
    finally:
      sys.path = sys_path_saved


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1:
    rebuild_tests(sys.argv[1])
  else:
    print("Usage: python agent_test_runner.py <folder>")
