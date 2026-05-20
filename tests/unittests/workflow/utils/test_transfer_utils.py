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

"""Tests for agent transfer utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow.utils._transfer_utils import resolve_and_derive_transfer_context
import pytest


def test_resolve_and_derive_transfer_context_raises_value_error_on_self_transfer():
  """resolve_and_derive_transfer_context raises ValueError when target is the current agent."""
  # Arrange
  current = LlmAgent(name='current')
  root = LlmAgent(name='root', sub_agents=[current])

  # Act & Assert
  with pytest.raises(ValueError, match='cannot transfer to itself'):
    resolve_and_derive_transfer_context(
        'current', current, root, MagicMock(), None
    )


def test_resolve_and_derive_transfer_context_returns_child_context():
  """resolve_and_derive_transfer_context returns current context as parent context for CHILD transfers."""
  # Arrange
  target = LlmAgent(name='target')
  current = LlmAgent(name='current', sub_agents=[target])
  root = LlmAgent(name='root', sub_agents=[current])

  curr_ctx = MagicMock()

  # Act
  resolved_agent, parent_ctx = resolve_and_derive_transfer_context(
      'target', current, root, curr_ctx, None
  )

  # Assert
  assert resolved_agent is target
  assert parent_ctx is curr_ctx


def test_resolve_and_derive_transfer_context_returns_sibling_context():
  """resolve_and_derive_transfer_context returns parent context for SIBLING transfers."""
  # Arrange
  current = LlmAgent(name='current')
  target = LlmAgent(name='target')
  root = LlmAgent(name='root', sub_agents=[current, target])

  parent_ctx = MagicMock()

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'target', current, root, MagicMock(), parent_ctx
  )

  # Assert
  assert resolved_agent is target
  assert derived_ctx is parent_ctx


def test_resolve_and_derive_transfer_context_climbs_parent_context():
  """resolve_and_derive_transfer_context climbs context chain to find the target parent's parent context."""
  # Arrange
  root_ctx = MagicMock()
  root_ctx.node = MagicMock()
  root_ctx.node.name = 'root'
  root_ctx.parent_ctx = MagicMock()
  root_ctx.parent_ctx.node = None
  root_ctx.parent_ctx.parent_ctx = None

  child_ctx = MagicMock()
  child_ctx.node = MagicMock()
  child_ctx.node.name = 'child'
  child_ctx.parent_ctx = root_ctx

  # Target is 'root', current is 'child'
  child = LlmAgent(name='child')
  root = LlmAgent(name='root', sub_agents=[child])
  child.parent_agent = root

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'root', child, root, child_ctx, None
  )

  # Assert
  assert resolved_agent is root
  assert derived_ctx is root_ctx.parent_ctx


def test_resolve_and_derive_transfer_context_returns_root_context_when_parent_bypassed():
  """resolve_and_derive_transfer_context returns root context for PARENT transfers when parent was bypassed."""
  # Arrange
  root_ctx = MagicMock()
  root_ctx.node = None
  root_ctx.parent_ctx = None

  child_ctx = MagicMock()
  child_ctx.node = MagicMock()
  child_ctx.node.name = 'child'
  child_ctx.parent_ctx = root_ctx

  # Target is 'root', current is 'child'
  child = LlmAgent(name='child')
  root = LlmAgent(name='root', sub_agents=[child])
  child.parent_agent = root

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'root', child, root, child_ctx, None
  )

  # Assert
  assert resolved_agent is root
  assert derived_ctx is root_ctx


def test_resolve_and_derive_transfer_context_returns_none_when_agent_not_found():
  """resolve_and_derive_transfer_context returns (None, None) when target agent is not found."""
  # Arrange
  current = LlmAgent(name='current')
  root = LlmAgent(name='root', sub_agents=[current])

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'target', current, root, MagicMock(), None
  )

  # Assert
  assert resolved_agent is None
  assert derived_ctx is None


def test_resolve_and_derive_transfer_context_returns_target_and_none_when_no_relationship():
  """resolve_and_derive_transfer_context returns (target_agent, None) for unrelated transfers."""
  # Arrange
  current = LlmAgent(name='current')
  target = LlmAgent(name='target')
  root1 = LlmAgent(name='root1', sub_agents=[current])
  root2 = LlmAgent(name='root2', sub_agents=[target])

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'target', current, root2, MagicMock(), None
  )

  # Assert
  assert resolved_agent is target
  assert derived_ctx is None


def test_resolve_and_derive_transfer_context_works_with_cloned_agents():
  """resolve_and_derive_transfer_context works correctly when the current agent is cloned (name-based matching)."""
  # Arrange
  target = LlmAgent(name='target')
  current = LlmAgent(name='current', sub_agents=[target])
  root = LlmAgent(name='root', sub_agents=[current])

  cloned_current = current.clone()
  assert cloned_current is not current
  assert cloned_current.name == current.name

  curr_ctx = MagicMock()

  # Act
  resolved_agent, derived_ctx = resolve_and_derive_transfer_context(
      'target', cloned_current, root, curr_ctx, None
  )

  # Assert
  assert resolved_agent is target
  assert derived_ctx is curr_ctx
