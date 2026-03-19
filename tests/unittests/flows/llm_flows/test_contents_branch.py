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

"""Tests for branch filtering in contents module.

Branch format: agent_1.agent_2.agent_3 (parent.child.grandchild)
Child agents can see parent agents' events, but not sibling agents' events.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.contents import request_processor
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_branch_filtering_child_sees_parent():
  """Test that child agents can see parent agents' events."""
  agent = Agent(model="gemini-2.5-flash", name="child_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set current branch as child of "parent_agent"
  invocation_context.branch = "parent_agent.child_agent"

  # Add events from parent and child levels
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("User message"),
      ),
      Event(
          invocation_id="inv2",
          author="parent_agent",
          content=types.ModelContent("Parent agent response"),
          branch="parent_agent",  # Parent branch - should be included
      ),
      Event(
          invocation_id="inv3",
          author="child_agent",
          content=types.ModelContent("Child agent response"),
          branch="parent_agent.child_agent",  # Current branch - should be included
      ),
      Event(
          invocation_id="inv4",
          author="child_agent",
          content=types.ModelContent("Excluded response 1"),
          branch="parent_agent.child_agent000",  # Prefix match BUT not itself/ancestor - should be excluded
      ),
      Event(
          invocation_id="inv5",
          author="child_agent",
          content=types.ModelContent("Excluded response 2"),
          branch="parent_agent.child",  # Prefix match BUT not itself/ancestor - should be excluded
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify child can see user message and parent events, but not sibling events
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0] == types.UserContent("User message")
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[parent_agent] said: Parent agent response"),
  ]
  assert llm_request.contents[2] == types.ModelContent("Child agent response")


@pytest.mark.asyncio
async def test_branch_filtering_excludes_sibling_agents():
  """Test that sibling agents cannot see each other's events."""
  agent = Agent(model="gemini-2.5-flash", name="child_agent1")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set current branch as first child
  invocation_context.branch = "parent_agent.child_agent1"

  # Add events from parent, current child, and sibling child
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("User message"),
      ),
      Event(
          invocation_id="inv2",
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch="parent_agent",  # Parent - should be included
      ),
      Event(
          invocation_id="inv3",
          author="child_agent1",
          content=types.ModelContent("Child1 response"),
          branch="parent_agent.child_agent1",  # Current - should be included
      ),
      Event(
          invocation_id="inv4",
          author="child_agent2",
          content=types.ModelContent("Sibling response"),
          branch="parent_agent.child_agent2",  # Sibling - should be excluded
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify sibling events are excluded, but parent and current agent events included
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0] == types.UserContent("User message")
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[parent_agent] said: Parent response"),
  ]
  assert llm_request.contents[2] == types.ModelContent("Child1 response")


@pytest.mark.asyncio
async def test_branch_filtering_no_branch_allows_all():
  """Test that events are included when no branches are set."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # No current branch set (None)
  invocation_context.branch = None

  # Add events with and without branches
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("No branch message"),
          branch=None,
      ),
      Event(
          invocation_id="inv2",
          author="agent1",
          content=types.ModelContent("Agent with branch"),
          branch="agent1",
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent("Another no branch"),
          branch=None,
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify all events are included when no current branch
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0] == types.UserContent("No branch message")
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[agent1] said: Agent with branch"),
  ]
  assert llm_request.contents[2] == types.UserContent("Another no branch")


@pytest.mark.asyncio
async def test_branch_filtering_grandchild_sees_grandparent():
  """Test that deeply nested child agents can see all ancestor events."""
  agent = Agent(model="gemini-2.5-flash", name="grandchild_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set deeply nested branch: grandparent.parent.grandchild
  invocation_context.branch = "grandparent_agent.parent_agent.grandchild_agent"

  # Add events from all levels of hierarchy
  events = [
      Event(
          invocation_id="inv1",
          author="grandparent_agent",
          content=types.ModelContent("Grandparent response"),
          branch="grandparent_agent",
      ),
      Event(
          invocation_id="inv2",
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch="grandparent_agent.parent_agent",
      ),
      Event(
          invocation_id="inv3",
          author="grandchild_agent",
          content=types.ModelContent("Grandchild response"),
          branch="grandparent_agent.parent_agent.grandchild_agent",
      ),
      Event(
          invocation_id="inv4",
          author="sibling_agent",
          content=types.ModelContent("Sibling response"),
          branch="grandparent_agent.parent_agent.sibling_agent",
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify only ancestors and current level are included
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[grandparent_agent] said: Grandparent response"),
  ]
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[parent_agent] said: Parent response"),
  ]
  assert llm_request.contents[2] == types.ModelContent("Grandchild response")


@pytest.mark.asyncio
async def test_branch_filtering_parent_cannot_see_child():
  """Test that parent agents cannot see child agents' events."""
  agent = Agent(model="gemini-2.5-flash", name="parent_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set current branch as parent
  invocation_context.branch = "parent_agent"

  # Add events from parent and its children
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("User message"),
      ),
      Event(
          invocation_id="inv2",
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch="parent_agent",
      ),
      Event(
          invocation_id="inv3",
          author="child_agent",
          content=types.ModelContent("Child response"),
          branch="parent_agent.child_agent",
      ),
      Event(
          invocation_id="inv4",
          author="grandchild_agent",
          content=types.ModelContent("Grandchild response"),
          branch="parent_agent.child_agent.grandchild_agent",
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify parent cannot see child or grandchild events
  assert llm_request.contents == [
      types.UserContent("User message"),
      types.ModelContent("Parent response"),
  ]
