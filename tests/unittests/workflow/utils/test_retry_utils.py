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

import random

from google.adk import platform as adk_platform
from google.adk.agents.base_agent import BaseAgent
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._errors import WorkflowConfigurationError
from google.adk.workflow._errors import WorkflowInvariantError
from google.adk.workflow._node import Node
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._parallel_worker import _ParallelWorker
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow.utils._retry_utils import _delegates_item_retry
from google.adk.workflow.utils._retry_utils import _get_retry_delay
from google.adk.workflow.utils._retry_utils import _has_own_retry_config
from google.adk.workflow.utils._retry_utils import _node_retry_config
from google.adk.workflow.utils._retry_utils import _should_retry_node
from google.adk.workflow.utils._retry_utils import _with_retry_config
import pytest


class _CustomValueError(ValueError):
  """A subclass of a built-in exception."""


class TestGetRetryDelay:

  def test_returns_default_delay_without_config(self):
    """Returns default delay of 1.0 second when config is missing."""
    state = NodeState(attempt_count=1)

    result = _get_retry_delay(None, state)

    assert result == 1.0

  def test_returns_initial_delay_on_first_failure(self):
    """Returns initial delay on the first failure attempt."""
    config = RetryConfig(initial_delay=2.0, jitter=0.0)
    state = NodeState(attempt_count=1)

    result = _get_retry_delay(config, state)

    assert result == 2.0

  def test_applies_exponential_backoff(self):
    """Applies exponential backoff for subsequent attempts."""
    config = RetryConfig(initial_delay=2.0, backoff_factor=2.0, jitter=0.0)
    state = NodeState(attempt_count=2)

    result = _get_retry_delay(config, state)

    assert result == 4.0

  def test_caps_at_max_delay(self):
    """Caps calculated delay at the specified maximum delay."""
    config = RetryConfig(
        initial_delay=2.0, backoff_factor=10.0, max_delay=15.0, jitter=0.0
    )
    state = NodeState(attempt_count=2)

    result = _get_retry_delay(config, state)

    assert result == 15.0

  def test_adds_jitter_when_enabled(self):
    """Adds random jitter to the calculated delay."""
    config = RetryConfig(initial_delay=10.0, backoff_factor=1.0, jitter=0.5)
    state = NodeState(attempt_count=1)

    delays = [_get_retry_delay(config, state) for _ in range(10)]

    assert all(5.0 <= d <= 15.0 for d in delays)
    assert len(set(delays)) > 1

  def test_jitter_stays_under_max_delay_without_bunching_on_it(self):
    """Keeps jittered delays under max_delay without piling them on the cap.

    Clamping the jittered delay to max_delay would respect the bound but land
    every overshooting draw on exactly max_delay, so retriers that reached the
    cap would all wake at the same instant.
    """
    config = RetryConfig(
        initial_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=1.0
    )
    state = NodeState(attempt_count=6)
    random.seed(20260807)

    delays = [_get_retry_delay(config, state) for _ in range(2000)]

    assert max(delays) <= 5.0
    at_cap = sum(1 for d in delays if d > 5.0 - 1e-9)
    assert at_cap / len(delays) < 0.01
    assert len(set(delays)) > 1

  def test_jitter_uses_platform_random_provider(self):
    """Jitter is drawn via the platform random seam so it is injectable.

    Frameworks that replay agent workflows (e.g. durable execution engines)
    install a deterministic random provider; the computed delay must then be
    reproducible across replays.
    """
    config = RetryConfig(initial_delay=10.0, backoff_factor=1.0, jitter=0.5)
    state = NodeState(attempt_count=1)
    rng = random.Random(42)
    adk_platform.set_random_provider(lambda: rng)
    try:
      expected_rng = random.Random(42)
      expected_delays = [
          max(0.0, 10.0 + expected_rng.uniform(-5.0, 5.0)) for _ in range(5)
      ]

      delays = [_get_retry_delay(config, state) for _ in range(5)]

      assert delays == expected_delays
    finally:
      adk_platform.reset_random_provider()


class TestShouldRetryNode:

  def test_no_config_never_retries(self):
    """Without a retry config, a node is never retried."""
    assert (
        _should_retry_node(RuntimeError(), None, NodeState(attempt_count=1))
        is False
    )

  @pytest.mark.parametrize("max_attempts", [0, 1])
  def test_max_attempts_zero_or_one_disables_retries(self, max_attempts):
    """max_attempts of 0 or 1 means no retries (per RetryConfig docs).

    A falsy-coalescing default (``max_attempts or 5``) wrongly treated an
    explicit ``0`` as unset and allowed 5 attempts.
    """
    config = RetryConfig(max_attempts=max_attempts)

    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=1))
        is False
    )

  def test_retries_until_max_attempts(self):
    """A node is retried while attempt_count is below max_attempts."""
    config = RetryConfig(max_attempts=3)

    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=1))
        is True
    )
    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=2))
        is True
    )
    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=3))
        is False
    )

  def test_unset_max_attempts_defaults_to_five(self):
    """When max_attempts is unset (None), the default of 5 applies."""
    config = RetryConfig()

    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=4))
        is True
    )
    assert (
        _should_retry_node(RuntimeError(), config, NodeState(attempt_count=5))
        is False
    )

  def test_retries_subclass_of_a_configured_exception(self):
    """A subclass of a configured exception is retried."""
    config = RetryConfig(exceptions=[ValueError])

    assert (
        _should_retry_node(
            _CustomValueError(), config, NodeState(attempt_count=1)
        )
        is True
    )

  def test_configuring_exception_retries_every_exception(self):
    """Configuring ``Exception`` retries anything deriving from it."""
    config = RetryConfig(exceptions=[Exception])

    assert (
        _should_retry_node(ValueError(), config, NodeState(attempt_count=1))
        is True
    )

  def test_unrelated_exception_is_not_retried(self):
    """An exception outside the configured hierarchy is not retried."""
    config = RetryConfig(exceptions=[ValueError])

    assert (
        _should_retry_node(KeyError(), config, NodeState(attempt_count=1))
        is False
    )

  def test_matching_survives_a_json_round_trip(self):
    """A config reloaded from JSON is equal to and matches like the original."""
    config = RetryConfig(exceptions=[ValueError, "KeyError"])

    reloaded = RetryConfig.model_validate_json(config.model_dump_json())

    assert reloaded == config
    assert (
        _should_retry_node(
            _CustomValueError(), reloaded, NodeState(attempt_count=1)
        )
        is True
    )

  @pytest.mark.parametrize(
      "error",
      [WorkflowConfigurationError("bad config"), WorkflowInvariantError("bug")],
  )
  def test_configuration_and_invariant_errors_are_never_retried(
      self, error: Exception
  ) -> None:
    """Deterministic configuration and invariant errors are never retried."""
    config = RetryConfig(
        max_attempts=3, exceptions=[Exception, ValueError, RuntimeError]
    )

    assert (
        _should_retry_node(error, config, NodeState(attempt_count=1)) is False
    )


class TestDelegatesItemRetry:

  @pytest.mark.parametrize(
      ("target_node", "expected"),
      [
          (_ParallelWorker(node=BaseNode(name="item")), True),
          (Node(name="parallel", parallel_worker=True), True),
          (Node(name="sequential", parallel_worker=False), False),
          (BaseNode(name="plain"), False),
      ],
      ids=["parallel_worker", "parallel_node", "sequential_node", "base_node"],
  )
  def test_delegates_item_retry(
      self, target_node: BaseNode, expected: bool
  ) -> None:
    """Returns True only for _ParallelWorker and Node(parallel_worker=True)."""
    assert _delegates_item_retry(target_node) is expected


class TestNodeRetryConfig:

  def test_returns_wrapped_node_policy_for_parallel_worker(self) -> None:
    """Returns the wrapped node's policy when worker.retry_config is None."""
    item_policy = RetryConfig(max_attempts=2)
    worker = _ParallelWorker(
        node=BaseNode(name="item"),
        retry_config=item_policy,
    )

    assert worker.retry_config is None
    assert _node_retry_config(worker) is item_policy

  def test_outer_worker_policy_wins_when_set_after_construction(self) -> None:
    """Returns worker.retry_config when set directly after construction."""
    item_policy = RetryConfig(max_attempts=2)
    outer_policy = RetryConfig(max_attempts=4)
    worker = _ParallelWorker(
        node=BaseNode(name="item"),
        retry_config=item_policy,
    )
    worker.retry_config = outer_policy

    assert _node_retry_config(worker) is outer_policy


class TestHasOwnRetryConfig:

  @pytest.mark.parametrize(
      ("target_node", "expected"),
      [
          (BaseNode(name="plain"), False),
          (
              BaseNode(
                  name="retried", retry_config=RetryConfig(max_attempts=2)
              ),
              True,
          ),
          (_ParallelWorker(node=BaseNode(name="item")), False),
          (
              _ParallelWorker(
                  node=BaseNode(name="item"),
                  retry_config=RetryConfig(max_attempts=2),
              ),
              True,
          ),
          (
              _ParallelWorker(
                  node=_ParallelWorker(
                      node=BaseNode(name="item"),
                      retry_config=RetryConfig(max_attempts=2),
                  )
              ),
              True,
          ),
      ],
      ids=[
          "base_node_without_retry",
          "base_node_with_retry",
          "parallel_worker_without_retry",
          "parallel_worker_with_retry",
          "nested_parallel_worker_with_retry",
      ],
  )
  def test_checks_node_and_parallel_worker_inner_node(
      self, target_node: BaseNode, expected: bool
  ) -> None:
    """Returns True only when a node has a retry_config of its own.

    For a ParallelWorker, the retry_config of its item node counts.
    """
    assert _has_own_retry_config(target_node) is expected


class TestWithRetryConfig:

  def test_copies_base_node_and_preserves_original(self) -> None:
    """Copies a BaseNode with the given retry_config without mutating it."""
    original = BaseNode(name="step")
    policy = RetryConfig(max_attempts=3)

    with_policy = _with_retry_config(original, policy)
    cleared = _with_retry_config(with_policy, None)

    assert original.retry_config is None
    assert with_policy is not original
    assert with_policy.retry_config == policy
    assert cleared.retry_config is None

  def test_clones_agent_and_preserves_parent_agent(self) -> None:
    """Clones a BaseAgent with retry_config while keeping parent_agent."""
    parent = BaseAgent(
        name="parent",
        sub_agents=[
            BaseAgent(name="child", sub_agents=[BaseAgent(name="grandchild")])
        ],
    )
    child = parent.sub_agents[0]
    policy = RetryConfig(max_attempts=2)

    cloned = _with_retry_config(child, policy)

    assert isinstance(cloned, BaseAgent)
    assert cloned is not child
    assert child.retry_config is None
    assert cloned.retry_config == policy
    assert cloned.parent_agent is parent
    assert cloned.sub_agents[0].parent_agent is cloned
