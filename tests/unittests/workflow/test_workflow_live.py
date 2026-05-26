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

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from . import testing_utils

# --- Mock Nodes and Agents for Testing Live Mode Design ---


class _MockNonLiveNode(BaseNode):
  """A standard non-live node whose signature does NOT accept live_request_queue."""

  called: bool = False
  actual_input: Any = None

  def __init__(self, *, name: str):
    super().__init__(name=name)

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    self.called = True
    self.actual_input = node_input
    yield Event(output=f"{self.name}_output")


class _ConstantNode(BaseNode):
  """A node that outputs a constant value."""

  output_value: Any = None

  def __init__(self, *, name: str, output_value: Any):
    super().__init__(name=name)
    self.output_value = output_value

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.output_value)


class _DynamicLiveSchedulerNode(BaseNode):
  """A node that dynamically schedules a child live node using ctx.run_node()."""

  child_node: BaseNode | None = None
  child_output: Any = None

  def __init__(self, *, name: str, child_node: BaseNode):
    super().__init__(name=name, rerun_on_resume=True)
    self.child_node = child_node

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.child_node:
      self.child_output = await ctx.run_node(
          self.child_node, node_input=node_input
      )
    yield Event(output=f"{self.name}_output")


# --- Live Workflow Unit Tests (TDD) ---


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        "when scheduler preserves originating node_input on resume."
    ),
)
@pytest.mark.asyncio
async def test_hybrid_live_non_live_nodes():
  """CUJ 1: A workflow has hybrid live & non-live nodes."""
  mock_model1 = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: node1_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: node1_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "LiveNode1_output"},
                  )
              ])
          ),
      ]
  )
  live_node1 = LlmAgent(
      name="LiveNode1",
      model=mock_model1,
      mode="task",
      instruction="Handle live interaction 1.",
  )
  non_live_node = _MockNonLiveNode(name="NonLiveNode")
  mock_model2 = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: node2_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: node2_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "LiveNode2_output"},
                  )
              ])
          ),
      ]
  )
  live_node2 = LlmAgent(
      name="LiveNode2",
      model=mock_model2,
      mode="task",
      instruction="Handle live interaction 2.",
  )

  wf = Workflow(
      name="hybrid_workflow",
      edges=[
          (START, live_node1),
          (live_node1, non_live_node),
          (non_live_node, live_node2),
      ],
  )

  live_queue = LiveRequestQueue()

  # Pre-seed first live node's requests
  live_queue.send_realtime(
      types.Blob(data=b"node1_start", mime_type="audio/pcm")
  )
  live_queue.send_realtime(types.Blob(data=b"node1_end", mime_type="audio/pcm"))

  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
      run_config=testing_utils.RunConfig(
          get_session_config={"state_delta": {"__START__": "start_input"}}
      ),
  ):
    events.append(event)
    if event.output == "NonLiveNode_output":
      # First live node and non-live node completed! Now feed the second live node's requests:
      live_queue.send_realtime(
          types.Blob(data=b"node2_start", mime_type="audio/pcm")
      )
      live_queue.send_realtime(
          types.Blob(data=b"node2_end", mime_type="audio/pcm")
      )

  # 1. Assert exact outputs sequence
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [
      {"result": "LiveNode1_output"},
      "NonLiveNode_output",
      {"result": "LiveNode2_output"},
  ]
  assert non_live_node.actual_input == {"result": "LiveNode1_output"}

  # 2. Assert intermediate content events (conversational turns)
  content_texts = [
      p.text
      for e in events
      if e.content and e.content.parts and e.output is None
      for p in e.content.parts
      if p.text
  ]
  assert content_texts == [
      "Acknowledged: node1_start",
      "Acknowledged: node1_end",
      "Acknowledged: node2_start",
      "Acknowledged: node2_end",
  ]

  # 3. Assert live requests fed to the models
  assert [b.data for b in mock_model1.live_blobs] == [
      b"node1_start",
      b"node1_end",
  ]
  assert [b.data for b in mock_model2.live_blobs] == [
      b"node2_start",
      b"node2_end",
  ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        "when scheduler preserves originating node_input on resume."
    ),
)
@pytest.mark.asyncio
async def test_nested_workflow_has_live_node():
  """CUJ 2: A nested workflow has a live node."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: inner_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: inner_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "InnerLiveNode_output"},
                  )
              ])
          ),
      ]
  )
  live_node = LlmAgent(
      name="InnerLiveNode",
      model=mock_model,
      mode="task",
      instruction="Handle inner live interaction.",
  )
  inner_wf = Workflow(name="inner_wf", edges=[(START, live_node)])
  outer_wf = Workflow(name="outer_wf", edges=[(START, inner_wf)])

  live_queue = LiveRequestQueue()
  live_queue.send_realtime(
      types.Blob(data=b"inner_start", mime_type="audio/pcm")
  )
  live_queue.send_realtime(types.Blob(data=b"inner_end", mime_type="audio/pcm"))

  ss = InMemorySessionService()
  runner = Runner(app_name=outer_wf.name, node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name=outer_wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
      run_config=testing_utils.RunConfig(
          get_session_config={"state_delta": {"__START__": "start_input"}}
      ),
  ):
    events.append(event)

  # Assert exact outputs sequence
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [{"result": "InnerLiveNode_output"}]

  # Assert content events
  content_texts = [
      p.text
      for e in events
      if e.content and e.content.parts and e.output is None
      for p in e.content.parts
      if p.text
  ]
  assert content_texts == [
      "Acknowledged: inner_start",
      "Acknowledged: inner_end",
  ]
  assert [b.data for b in mock_model.live_blobs] == [
      b"inner_start",
      b"inner_end",
  ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        "when scheduler preserves originating node_input on resume."
    ),
)
@pytest.mark.asyncio
async def test_nested_live_node_and_outer_live_node():
  """CUJ 3: A nested workflow has live node & outer workflow then has a live node."""
  mock_model_inner = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: inner_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: inner_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "InnerLiveNode_output"},
                  )
              ])
          ),
      ]
  )
  inner_live = LlmAgent(
      name="InnerLiveNode",
      model=mock_model_inner,
      mode="task",
      instruction="Handle inner live interaction.",
  )
  inner_wf = Workflow(name="inner_wf", edges=[(START, inner_live)])

  mock_model_outer = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: outer_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: outer_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "OuterLiveNode_output"},
                  )
              ])
          ),
      ]
  )
  outer_live = LlmAgent(
      name="OuterLiveNode",
      model=mock_model_outer,
      mode="task",
      instruction="Handle outer live interaction.",
  )
  prep_node = _ConstantNode(name="PrepNode", output_value="prep_output")

  wf = Workflow(
      name="nested_sequential_live",
      edges=[
          (START, inner_wf),
          (inner_wf, prep_node),
          (prep_node, outer_live),
      ],
  )

  live_queue = LiveRequestQueue()
  live_queue.send_realtime(
      types.Blob(data=b"inner_start", mime_type="audio/pcm")
  )
  live_queue.send_realtime(types.Blob(data=b"inner_end", mime_type="audio/pcm"))

  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
      run_config=testing_utils.RunConfig(
          get_session_config={"state_delta": {"__START__": "start_input"}}
      ),
  ):
    events.append(event)
    if event.output == "prep_output":
      # Inner live node and prep node completed! Feed outer node's requests:
      live_queue.send_realtime(
          types.Blob(data=b"outer_start", mime_type="audio/pcm")
      )
      live_queue.send_realtime(
          types.Blob(data=b"outer_end", mime_type="audio/pcm")
      )

  # Assert exact outputs sequence
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [
      {"result": "InnerLiveNode_output"},
      "prep_output",
      {"result": "OuterLiveNode_output"},
  ]

  # Assert content events
  content_texts = [
      p.text
      for e in events
      if e.content and e.content.parts and e.output is None
      for p in e.content.parts
      if p.text
  ]
  assert content_texts == [
      "Acknowledged: inner_start",
      "Acknowledged: inner_end",
      "Acknowledged: outer_start",
      "Acknowledged: outer_end",
  ]
  assert [b.data for b in mock_model_inner.live_blobs] == [
      b"inner_start",
      b"inner_end",
  ]
  assert [b.data for b in mock_model_outer.live_blobs] == [
      b"outer_start",
      b"outer_end",
  ]


@pytest.mark.asyncio
async def test_dynamic_node_scheduling_of_live_node():
  """CUJ 4: A node in workflow dynamically schedules a live node using ctx.run_node()."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: dynamic_start")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent(
                  [types.Part.from_text(text="Acknowledged: dynamic_end")]
              )
          ),
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "DynamicLiveNode_output"},
                  )
              ])
          ),
      ]
  )
  live_node = LlmAgent(
      name="DynamicLiveNode",
      model=mock_model,
      mode="task",
      instruction="Handle dynamic live interaction.",
  )
  scheduler_node = _DynamicLiveSchedulerNode(
      name="SchedulerNode", child_node=live_node
  )

  wf = Workflow(name="dynamic_wf", edges=[(START, scheduler_node)])

  live_queue = LiveRequestQueue()
  live_queue.send_realtime(
      types.Blob(data=b"dynamic_start", mime_type="audio/pcm")
  )
  live_queue.send_realtime(
      types.Blob(data=b"dynamic_end", mime_type="audio/pcm")
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
      run_config=testing_utils.RunConfig(
          get_session_config={"state_delta": {"__START__": "start_input"}}
      ),
  ):
    events.append(event)

  # Assert exact outputs sequence
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [
      {"result": "DynamicLiveNode_output"},
      "SchedulerNode_output",
  ]
  assert scheduler_node.child_output == {"result": "DynamicLiveNode_output"}

  # Assert content events
  content_texts = [
      p.text
      for e in events
      if e.content and e.content.parts and e.output is None
      for p in e.content.parts
      if p.text
  ]
  assert content_texts == [
      "Acknowledged: dynamic_start",
      "Acknowledged: dynamic_end",
  ]
  assert [b.data for b in mock_model.live_blobs] == [
      b"dynamic_start",
      b"dynamic_end",
  ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        "when scheduler preserves originating node_input on resume."
    ),
)
@pytest.mark.asyncio
async def test_live_node_output_passed_to_downstream():
  """CUJ 5: Dedicated test verifying output of a live node is passed to the next node."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=testing_utils.ModelContent([
                  types.Part.from_function_call(
                      name="finish_task",
                      args={"result": "LiveNode_output"},
                  )
              ])
          ),
      ]
  )
  live_node = LlmAgent(
      name="LiveNode",
      model=mock_model,
      mode="task",
      instruction="Handle live interaction.",
  )
  non_live_node = _MockNonLiveNode(name="NonLiveNode")

  wf = Workflow(
      name="dataflow_workflow",
      edges=[
          (START, live_node),
          (live_node, non_live_node),
      ],
  )

  live_queue = LiveRequestQueue()
  live_queue.send_realtime(types.Blob(data=b"start_msg", mime_type="audio/pcm"))
  live_queue.send_realtime(types.Blob(data=b"end_msg", mime_type="audio/pcm"))

  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
      run_config=testing_utils.RunConfig(
          get_session_config={"state_delta": {"__START__": "start_input"}}
      ),
  ):
    events.append(event)

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [{"result": "LiveNode_output"}, "NonLiveNode_output"]
  assert non_live_node.actual_input == {
      "result": "LiveNode_output"
  }, "The downstream node must receive the live node's exact output"
  assert [b.data for b in mock_model.live_blobs] == [b"start_msg", b"end_msg"]


@pytest.mark.asyncio
async def test_single_turn_agent_runs_as_non_live_in_live_session():
  """CUJ 6: A single_turn LlmAgent in a live session runs as non-live and consumes node_input."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          "SingleTurn_output",
      ]
  )
  prep_node = _ConstantNode(
      name="ConstantNode", output_value="initial_text_input"
  )
  single_turn_node = LlmAgent(
      name="SingleTurnNode",
      model=mock_model,
      mode="single_turn",
      instruction="Summarize the input.",
  )
  capture = _MockNonLiveNode(name="capture")

  wf = Workflow(
      name="single_turn_live_wf",
      edges=[
          (START, prep_node),
          (prep_node, single_turn_node),
          (single_turn_node, capture),
      ],
  )

  live_queue = LiveRequestQueue()
  live_queue.send_realtime(
      types.Blob(data=b"ignored_audio", mime_type="audio/pcm")
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id="u")

  events = []
  async for event in runner.run_live(
      user_id="u",
      session_id=session.id,
      live_request_queue=live_queue,
  ):
    events.append(event)

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ["initial_text_input", "capture_output"]
  assert capture.actual_input == "SingleTurn_output"
  # Verify that the model received the initial_text_input (node_input) and NOT the live queue audio
  assert len(mock_model.requests) == 1
  assert (
      mock_model.requests[0].contents[0].parts[0].text == "initial_text_input"
  )
  assert mock_model.live_blobs == []
