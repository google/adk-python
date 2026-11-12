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

import abc
from collections.abc import AsyncGenerator
from collections.abc import Iterator
from typing import Any
from typing import final
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from ..utils._schema_utils import SchemaType
from ..utils._schema_utils import validate_node_data
from ..utils._sync_runner import _use_sync_callable_context
from ._errors import WorkflowConfigurationError
from ._retry_config import RetryConfig

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..events.event import Event
  from ._graph import Edge


# The `abc.ABC` base is load-bearing and must not be dropped. Subclasses
# declare `@abc.abstractmethod` without inheriting `abc.ABC` themselves, relying
# on this class for the metaclass. Static type checkers do not see `ABCMeta`
# through pydantic's `ModelMetaclass`, so without an explicit base they treat
# those subclasses as concrete and report the abstract methods as returning
# `None`.
class BaseNode(BaseModel, abc.ABC):
  """A base class for all nodes in the workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(...)
  """The unique name of the node within the workflow graph."""

  @field_validator('name')
  @classmethod
  def _validate_name(cls, v: str) -> str:
    # Pydantic re-wraps this as a ValidationError, so the class below records
    # whose mistake it is rather than being catchable on its own.
    if not v.isidentifier():
      raise WorkflowConfigurationError(
          f"Node name '{v}' must be a valid Python identifier."
      )
    return v

  description: str = ''
  """A human-readable description of what this node does."""

  rerun_on_resume: bool = False
  """Controls behavior when resuming after an interrupt.

  If True, the node reruns from scratch. If False, it completes immediately
  using the user's resuming input as the node's output.

  A rerun node that raised several interrupts reruns as soon as any one of them
  is answered, with the answers received so far in ``ctx.resume_inputs``. To
  keep waiting for the rest, yield their interrupts again.
  """

  wait_for_output: bool = False
  """If True, node only transitions to COMPLETED upon yielding output or route.

  Without output/route, the node enters WAITING state and downstream nodes are
  not triggered, allowing predecessors to re-trigger it. This is useful for nodes
  like ``JoinNode`` that run multiple times before producing a final output.

  WARNING: Completing execution without ever yielding output/route causes an
  indefinite WAITING state (deadlock). This is considered a user configuration error.
  """

  retry_config: RetryConfig | None = None
  """Configuration for retrying the node on failure.

  If set, failures of the node will trigger retries according to the
  specified policy. A retry after the node's own error runs it again in
  full, including the children it started with ``ctx.run_node()``. A retry
  after one of those children failed replays the children that the failed
  attempt had completed, at any depth below the node, instead of running
  them again.

  On a ``Workflow``, a failure of any node inside it is a failure of the
  workflow, so the whole sub-workflow is retried. Children that already
  produced an output or a state change are replayed rather than run again;
  a child that produced neither leaves nothing to replay and runs again.

  On a parallel worker (``parallel_worker=True``), the policy applies to each
  item: a failed item is retried in place while the other items keep
  running, and the worker itself is not retried for an item's failure. The
  policy does not cover a failure of the worker as a whole, such as its
  ``timeout``, except on a ``Node`` subclass with ``parallel_worker=True``,
  which keeps the policy on the worker as well.

  ``None`` means the node is not retried (the default).
  """

  timeout: float | None = None
  """Maximum time in seconds for this node to complete.

  If the node does not finish within this duration, it is cancelled and
  treated as a failure (raising ``NodeTimeoutError``).  This integrates
  with ``retry_config`` — a timed-out node can be retried if retries
  are configured.

  ``None`` means no timeout (the node runs until completion).
  """

  input_schema: SchemaType | None = None
  """Schema to validate and coerce node input data.

  Validated with ``TypeAdapter``. A raw ``dict`` JSON schema or a genai
  ``Schema`` is accepted but never enforced.

  ``None`` means no input validation (the default). ``FunctionNode`` fills
  this in after construction from the wrapped function's type hints, so on
  that subclass ``None`` also means inference found nothing usable.
  """

  output_schema: SchemaType | None = None
  """Schema to validate and coerce node output data.

  Validated with ``TypeAdapter``; a validated ``BaseModel`` is dumped to a
  dict with ``None`` fields dropped. A raw ``dict`` JSON schema or a genai
  ``Schema`` is accepted but never enforced.

  ``None`` means no output validation (the default). ``FunctionNode`` fills
  this in after construction from the wrapped function's return hint, so on
  that subclass ``None`` also means inference found nothing usable.
  """

  state_schema: type[BaseModel] | None = None
  """Optional Pydantic model declaring the expected state keys and types.

  When set, ``ctx.state`` mutations are validated at runtime against
  this schema.  Child nodes inherit the schema from their parent
  (via InvocationContext) unless they declare their own.

  Prefixed keys (``app:``, ``user:``, ``temp:``) bypass validation.
  """

  def _validate_schema(self, data: Any, schema: Any) -> Any:
    """Validates data against a schema using validate_node_data helper."""
    return validate_node_data(schema, data)

  def _validate_input_data(self, data: Any) -> Any:
    """Validates data against input_schema if set."""
    return validate_node_data(self.input_schema, data, preserve_content=False)

  def _validate_output_data(self, data: Any) -> Any:
    """Validates data against output_schema if set."""
    return validate_node_data(self.output_schema, data, preserve_content=False)

  @staticmethod
  def _to_serializable(data: Any) -> Any:
    """Converts BaseModel instances to dicts recursively."""
    if isinstance(data, BaseModel):
      return data.model_dump()
    if isinstance(data, list):
      return [BaseNode._to_serializable(item) for item in data]
    if isinstance(data, dict):
      return {k: BaseNode._to_serializable(v) for k, v in data.items()}
    return data

  @final
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Event, None]:
    """Public entry point. Calls _run_impl, normalizes yields to Event.

    ``node_input`` is passed through ``_validate_input_data`` before
    ``_run_impl`` sees it, and every yielded output through
    ``_validate_output_data``.

    Normalization rules:
    - None -> skipped
    - Event -> pass through
    - RequestInput -> convert to interrupt Event
    - Any other value -> Event(output=value)
    """
    from ..events.event import Event
    from ..events.request_input import RequestInput
    from ..utils.context_utils import Aclosing

    _sentinel = object()
    node_input = self._validate_input_data(node_input)
    async with Aclosing(self._run_impl(ctx=ctx, node_input=node_input)) as agen:
      while True:
        # Bind per step rather than across yield so async-generator suspension
        # never leaks or cross-resets ContextVar tokens in the caller task.
        with _use_sync_callable_context(ctx):
          item = await anext(agen, _sentinel)
        if item is _sentinel:
          break
        if item is None:
          continue
        if isinstance(item, Event):
          if item.output is not None:
            item.output = self._validate_output_data(item.output)
          yield item
        elif isinstance(item, RequestInput):
          from .utils._workflow_hitl_utils import create_request_input_event

          yield create_request_input_event(item)
        else:
          validated = self._validate_output_data(item)
          yield Event(output=validated)

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Override point for node execution logic.

    Yields any of: Event, RequestInput, raw data, or None.
    BaseNode.run() normalizes all yields to Event before the caller
    sees them.

    The base implementation raises ``NotImplementedError``, but because this
    is an async generator it does so on the first iteration rather than at the
    call. ``BaseNode`` therefore stays constructible without an override,
    which ``START`` relies on.
    """
    raise NotImplementedError(
        f'_run_impl for {type(self).__name__} is not implemented.'
    )
    yield  # AsyncGenerator requires at least one yield statement

  def _iter_static_children(self) -> Iterator[BaseNode]:
    """Yields direct static children declared on this node.

    Private helper for internal ADK framework use only. Do not call or override
    outside ADK.

    A static child is declared on the node structure and runs within the same
    application context (same Runner, session, and plugins). Nodes created
    dynamically at runtime or executed in an isolated runner (e.g.
    ``AgentTool``) are not static children.
    """
    return
    yield

  @property
  def _requires_all_predecessors(self) -> bool:
    """If True, the node waits for all predecessors to complete before running."""
    return False

  def _validate_outgoing_edges(self, edges: list[Edge]) -> None:
    """Checks the node's outgoing edges when its graph is validated.

    Override to reject wiring the node cannot serve, such as routes it never
    emits, by raising ``GraphValidationError``. The default accepts any edges.

    Args:
      edges: Every edge in the graph whose ``from_node`` is this node.
    """


def _walk_static_nodes(root: BaseNode) -> Iterator[BaseNode]:
  """Performs an iterative pre-order depth-first traversal of static nodes.

  Private helper for internal ADK framework use only.

  Yields ``root`` first, then each reachable static child discovered via
  ``BaseNode._iter_static_children()``. Tracks visited nodes by identity
  (``id()``) to avoid duplicate visits in directed acyclic graphs and prevent
  infinite loops on cyclic references.
  """
  visited: set[int] = set()
  stack: list[BaseNode] = [root]
  while stack:
    curr = stack.pop()
    curr_id = id(curr)
    if curr_id in visited:
      continue
    visited.add(curr_id)
    yield curr
    children = list(curr._iter_static_children())
    stack.extend(reversed(children))


def find_static_node_path(root: BaseNode, target: BaseNode) -> str | None:
  """Returns the static (run-id-free) path of ``target`` within ``root``.

  The path is the chain of node names from ``root`` down to ``target``, joined
  by ``/`` and carrying no run ids, so it identifies a node's position in the
  tree independently of any particular run. Returns ``None`` when ``target`` is
  not reachable from ``root``.

  Children are discovered via ``_iter_static_children()`` as well as the node's
  Pydantic fields, so nodes held in a list or dict field are traversed as well.
  """
  visited: set[int] = set()

  def _recurse(curr: BaseNode) -> list[str] | None:
    if id(curr) in visited:
      return None
    visited.add(id(curr))

    if curr is target:
      return [curr.name]

    for child in curr._iter_static_children():
      path = _recurse(child)
      if path:
        return [curr.name] + path

    for _, val in curr:
      if isinstance(val, BaseNode):
        path = _recurse(val)
        if path:
          return [curr.name] + path
      elif isinstance(val, list):
        for item in val:
          if isinstance(item, BaseNode):
            path = _recurse(item)
            if path:
              return [curr.name] + path
      elif isinstance(val, dict):
        for item in val.values():
          if isinstance(item, BaseNode):
            path = _recurse(item)
            if path:
              return [curr.name] + path
    return None

  path_list = _recurse(root)
  if path_list:
    return '/'.join(path_list)
  return None


START = BaseNode(name='__START__')
"""Sentinel node marking the entry point of a workflow graph.

START is never executed. ``Workflow._seed_start_triggers`` records the
workflow's input and branch under START's name, then seeds triggers for
its successors directly.
"""
