# Copyright 2025 Google LLC
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

from google.adk.agents.branch import Branch
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
  # Set current branch as child - child has tokens {1, 2} (inherited 1 from parent, got 2 from fork)
  invocation_context.branch = Branch(tokens=frozenset({1, 2}))

  # Add events from parent and child levels
  # Using same invocation_id for all events to test branch filtering within invocation
  inv_id = invocation_context.invocation_id
  events = [
      Event(
          invocation_id=inv_id,
          author="user",
          content=types.UserContent("User message"),
          branch=Branch(),  # Root branch - visible to all
      ),
      Event(
          invocation_id=inv_id,
          author="parent_agent",
          content=types.ModelContent("Parent agent response"),
          branch=Branch(tokens=frozenset({1})),  # Parent branch - should be included ({1} ⊆ {1,2})
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent",
          content=types.ModelContent("Child agent response"),
          branch=Branch(tokens=frozenset({1, 2})),  # Current branch - should be included
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent",
          content=types.ModelContent("Excluded response 1"),
          branch=Branch(tokens=frozenset({1, 3})),  # Sibling branch - should be excluded ({1,3} ⊄ {1,2})
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent",
          content=types.ModelContent("Excluded response 2"),
          branch=Branch(tokens=frozenset({3})),  # Different branch - should be excluded ({3} ⊄ {1,2})
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
  # Set current branch as first child - has tokens {1, 2} (inherited 1 from parent, got 2 from fork)
  invocation_context.branch = Branch(tokens=frozenset({1, 2}))

  # Add events from parent, current child, and sibling child
  inv_id = invocation_context.invocation_id
  events = [
      Event(
          invocation_id=inv_id,
          author="user",
          content=types.UserContent("User message"),
          branch=Branch(),  # Root - visible to all
      ),
      Event(
          invocation_id=inv_id,
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch=Branch(tokens=frozenset({1})),  # Parent - should be included ({1} ⊆ {1,2})
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent1",
          content=types.ModelContent("Child1 response"),
          branch=Branch(tokens=frozenset({1, 2})),  # Current - should be included
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent2",
          content=types.ModelContent("Sibling response"),
          branch=Branch(tokens=frozenset({1, 3})),  # Sibling - should be excluded ({1,3} ⊄ {1,2})
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
  # Root branch (empty tokens) - can see all events
  invocation_context.branch = Branch()

  # Add events with various branches
  inv_id = invocation_context.invocation_id
  events = [
      Event(
          invocation_id=inv_id,
          author="user",
          content=types.UserContent("No branch message"),
          branch=Branch(),  # Root - visible
      ),
      Event(
          invocation_id=inv_id,
          author="agent1",
          content=types.ModelContent("Agent with branch"),
          branch=Branch(tokens=frozenset({1})),  # Not visible ({1} ⊄ {})
      ),
      Event(
          invocation_id=inv_id,
          author="user",
          content=types.UserContent("Another no branch"),
          branch=Branch(),  # Root - visible
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify only root events are visible (root can't see events with tokens)
  assert len(llm_request.contents) == 2
  assert llm_request.contents[0] == types.UserContent("No branch message")
  assert llm_request.contents[1] == types.UserContent("Another no branch")


@pytest.mark.asyncio
async def test_branch_filtering_grandchild_sees_grandparent():
  """Test that deeply nested child agents can see all ancestor events."""
  agent = Agent(model="gemini-2.5-flash", name="grandchild_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set deeply nested branch: grandchild has tokens {1, 2, 3}
  # (inherited 1 from grandparent, 2 from parent, got 3 from its own fork)
  invocation_context.branch = Branch(tokens=frozenset({1, 2, 3}))

  # Add events from all levels of hierarchy
  inv_id = invocation_context.invocation_id
  events = [
      Event(
          invocation_id=inv_id,
          author="grandparent_agent",
          content=types.ModelContent("Grandparent response"),
          branch=Branch(tokens=frozenset({1})),  # Should be visible ({1} ⊆ {1,2,3})
      ),
      Event(
          invocation_id=inv_id,
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch=Branch(tokens=frozenset({1, 2})),  # Should be visible ({1,2} ⊆ {1,2,3})
      ),
      Event(
          invocation_id=inv_id,
          author="grandchild_agent",
          content=types.ModelContent("Grandchild response"),
          branch=Branch(tokens=frozenset({1, 2, 3})),  # Should be visible (same)
      ),
      Event(
          invocation_id=inv_id,
          author="sibling_agent",
          content=types.ModelContent("Sibling response"),
          branch=Branch(tokens=frozenset({1, 2, 4})),  # Should be excluded ({1,2,4} ⊄ {1,2,3})
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
  # Set current branch as parent with token {1}
  invocation_context.branch = Branch(tokens=frozenset({1}))

  # Add events from parent and its children
  inv_id = invocation_context.invocation_id
  events = [
      Event(
          invocation_id=inv_id,
          author="user",
          content=types.UserContent("User message"),
          branch=Branch(),  # Root - visible to all
      ),
      Event(
          invocation_id=inv_id,
          author="parent_agent",
          content=types.ModelContent("Parent response"),
          branch=Branch(tokens=frozenset({1})),  # Should be visible (same)
      ),
      Event(
          invocation_id=inv_id,
          author="child_agent",
          content=types.ModelContent("Child response"),
          branch=Branch(tokens=frozenset({1, 2})),  # Should be excluded ({1,2} ⊄ {1})
      ),
      Event(
          invocation_id=inv_id,
          author="grandchild_agent",
          content=types.ModelContent("Grandchild response"),
          branch=Branch(tokens=frozenset({1, 2, 3})),  # Should be excluded ({1,2,3} ⊄ {1})
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
