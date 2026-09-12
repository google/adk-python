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

"""Validated, persisted graph documents for the visual graph workbench."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tempfile

from pydantic import BaseModel
from pydantic import Field

from .inspector import GraphEdge
from .inspector import GraphNode
from .inspector import GraphTopology

_EDITABLE_AGENT_TYPES = frozenset(
    {"llm_agent", "sequential", "parallel", "loop"}
)
_CONTAINER_TYPES = frozenset({"llm_agent", "sequential", "parallel", "loop"})


class GraphDocument(BaseModel):
  """A durable graph draft with optimistic-concurrency metadata."""

  schema_version: int = 1
  revision: int = Field(default=0, ge=0)
  source_digest: str | None = None
  topology: GraphTopology
  positions: dict[str, tuple[float, float]] = Field(default_factory=dict)


class GraphDocumentConflictError(RuntimeError):
  """Raised when a client attempts to overwrite a newer graph draft."""


class GraphDocumentStore:
  """Persists graph drafts beside an agent file with revision checking."""

  def __init__(self, *, agent_file_path: Path | None):
    self._agent_file_path = agent_file_path
    self._draft_path = (
        agent_file_path.parent / ".adk" / "graph-draft.json"
        if agent_file_path
        else None
    )

  def load(self, *, topology: GraphTopology) -> GraphDocument:
    """Loads a compatible draft or initializes one from inspected topology."""
    source_digest = self._source_digest()
    if not self._draft_path or not self._draft_path.is_file():
      return GraphDocument(topology=topology, source_digest=source_digest)

    try:
      document = GraphDocument.model_validate_json(self._draft_path.read_text())
    except (OSError, ValueError):
      return GraphDocument(topology=topology, source_digest=source_digest)

    if document.source_digest != source_digest:
      return GraphDocument(topology=topology, source_digest=source_digest)
    # An empty persisted draft cannot describe a non-empty source graph.  This
    # can happen after an interrupted first save; retaining it would leave the
    # workbench permanently blank and make Preview generate a fake root agent.
    if topology.nodes and not document.topology.nodes:
      return GraphDocument(topology=topology, source_digest=source_digest)
    return document

  def save(
      self,
      *,
      document: GraphDocument,
      expected_revision: int,
  ) -> GraphDocument:
    """Atomically saves a newer revision of a valid document."""
    current = self.load(topology=document.topology)
    if current.revision != expected_revision:
      raise GraphDocumentConflictError(
          "This graph has changed in another browser. Reload before saving."
      )

    saved = document.model_copy(
        update={
            "revision": expected_revision + 1,
            "source_digest": self._source_digest(),
        }
    )
    self._write_document(saved)
    return saved

  def update_source_digest(self, *, document: GraphDocument) -> GraphDocument:
    """Records a successful source write without changing the draft revision."""
    updated = document.model_copy(
        update={"source_digest": self._source_digest()}
    )
    self._write_document(updated)
    return updated

  def _write_document(self, document: GraphDocument) -> None:
    if not self._draft_path:
      return
    self._draft_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=self._draft_path.parent,
        delete=False,
    ) as temporary_file:
      temporary_file.write(document.model_dump_json(indent=2))
      temporary_path = Path(temporary_file.name)
    temporary_path.replace(self._draft_path)

  def _source_digest(self) -> str | None:
    if not self._agent_file_path or not self._agent_file_path.is_file():
      return None
    return hashlib.sha256(self._agent_file_path.read_bytes()).hexdigest()


def validate_topology(topology: GraphTopology) -> None:
  """Ensures a graph can be represented by the managed ADK source format."""
  nodes = {node.id: node for node in topology.nodes}
  if len(nodes) != len(topology.nodes):
    raise ValueError("Each graph node must have a unique id.")
  if topology.nodes and topology.root_id not in nodes:
    raise ValueError("root_id must identify a graph node.")
  if not topology.nodes and topology.root_id:
    raise ValueError("An empty graph must not have a root_id.")

  parent_counts: dict[str, int] = {node_id: 0 for node_id in nodes}
  tool_counts: dict[str, int] = {node_id: 0 for node_id in nodes}
  adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
  for edge in topology.edges:
    _validate_edge(edge=edge, nodes=nodes)
    if edge.type == "sub_agent":
      parent_counts[edge.target] += 1
      adjacency[edge.source].append(edge.target)
    else:
      tool_counts[edge.source] += 1

  for node in nodes.values():
    if node.type == "tool":
      if tool_counts[node.id] != 1:
        raise ValueError(
            f"Tool {node.label!r} must be bound to exactly one LLM agent."
        )
      continue
    if node.type not in _EDITABLE_AGENT_TYPES:
      raise ValueError(
          f"{node.label!r} uses unsupported type {node.type!r}. "
          "Custom agents are read-only in the graph editor."
      )
    if node.id == topology.root_id:
      if parent_counts[node.id]:
        raise ValueError("The root agent cannot be a sub-agent.")
    elif parent_counts[node.id] != 1:
      raise ValueError(
          f"Agent {node.label!r} must have exactly one parent connection."
      )

  _validate_acyclic(adjacency)


def generate_source(topology: GraphTopology) -> str:
  """Generates syntactically validated Python for a managed ADK graph."""
  validate_topology(topology)
  if not topology.nodes:
    return (
        "from google.adk.agents import LlmAgent\n\nroot_agent ="
        " LlmAgent(name='root')\n"
    )

  nodes = {node.id: node for node in topology.nodes}
  children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
  tools: dict[str, list[str]] = {node_id: [] for node_id in nodes}
  for edge in topology.edges:
    if edge.type == "sub_agent":
      children[edge.source].append(edge.target)
    else:
      tools[edge.target].append(edge.source)

  lines = [
      "from google.adk.agents import LlmAgent",
      "from google.adk.agents import LoopAgent",
      "from google.adk.agents import ParallelAgent",
      "from google.adk.agents import SequentialAgent",
      "",
  ]
  for node in nodes.values():
    if node.type == "tool":
      lines.extend(_tool_definition(node))

  for node_id in reversed(_agent_order(topology)):
    node = nodes[node_id]
    lines.extend(
        _agent_definition(
            node=node,
            children=[nodes[child] for child in children[node_id]],
            tools=[nodes[tool] for tool in tools[node_id]],
        )
    )
  lines.append(f"root_agent = {_identifier(nodes[topology.root_id])}")
  source = "\n".join(lines) + "\n"
  ast.parse(source)
  return source


def _validate_edge(*, edge: GraphEdge, nodes: dict[str, GraphNode]) -> None:
  if edge.source not in nodes or edge.target not in nodes:
    raise ValueError(f"Connection {edge.id!r} references a missing node.")
  source = nodes[edge.source]
  target = nodes[edge.target]
  if edge.type == "sub_agent":
    if source.type not in _CONTAINER_TYPES or target.type == "tool":
      raise ValueError("A sub-agent connection must link an agent to an agent.")
  elif edge.type == "tool_binding":
    if source.type != "tool" or target.type != "llm_agent":
      raise ValueError("A tool connection must link a tool to an LLM agent.")
  else:
    raise ValueError(f"Unsupported connection type {edge.type!r}.")


def _validate_acyclic(adjacency: dict[str, list[str]]) -> None:
  visiting: set[str] = set()
  visited: set[str] = set()

  def visit(node_id: str) -> None:
    if node_id in visiting:
      raise ValueError("Sub-agent connections cannot contain a cycle.")
    if node_id in visited:
      return
    visiting.add(node_id)
    for child_id in adjacency[node_id]:
      visit(child_id)
    visiting.remove(node_id)
    visited.add(node_id)

  for node_id in adjacency:
    visit(node_id)


def _agent_order(topology: GraphTopology) -> list[str]:
  children: dict[str, list[str]] = {node.id: [] for node in topology.nodes}
  for edge in topology.edges:
    if edge.type == "sub_agent":
      children[edge.source].append(edge.target)
  order: list[str] = []

  def visit(node_id: str) -> None:
    order.append(node_id)
    for child_id in children[node_id]:
      visit(child_id)

  if topology.root_id:
    visit(topology.root_id)
  return order


def _agent_definition(
    *, node: GraphNode, children: list[GraphNode], tools: list[GraphNode]
) -> list[str]:
  identifier = _identifier(node)
  description = node.description or ""
  if node.type == "llm_agent":
    kwargs = [f"name={node.label!r}", f"description={description!r}"]
    instruction = node.config.get("instruction", "")
    if isinstance(instruction, str) and instruction:
      kwargs.append(f"instruction={instruction!r}")
    model = node.config.get("model", "")
    if isinstance(model, str) and model:
      kwargs.append(f"model={model!r}")
    if children:
      kwargs.append(
          "sub_agents=["
          + ", ".join(_identifier(child) for child in children)
          + "]"
      )
    if tools:
      kwargs.append(
          "tools=[" + ", ".join(_identifier(tool) for tool in tools) + "]"
      )
    return [
        f"{identifier} = LlmAgent(",
        *[f"    {arg}," for arg in kwargs],
        ")",
        "",
    ]

  class_name = {
      "sequential": "SequentialAgent",
      "parallel": "ParallelAgent",
      "loop": "LoopAgent",
  }[node.type]
  kwargs = [f"name={node.label!r}", f"description={description!r}"]
  kwargs.append(
      "sub_agents=[" + ", ".join(_identifier(child) for child in children) + "]"
  )
  if node.type == "loop":
    max_iterations = node.config.get("max_iterations")
    if isinstance(max_iterations, int) and max_iterations > 0:
      kwargs.append(f"max_iterations={max_iterations}")
  return [
      f"{identifier} = {class_name}(",
      *[f"    {arg}," for arg in kwargs],
      ")",
      "",
  ]


def _tool_definition(node: GraphNode) -> list[str]:
  identifier = _identifier(node)
  implementation = node.config.get("implementation")
  if not isinstance(implementation, str) or not implementation.strip():
    raise ValueError(
        f"Tool {node.label!r} needs a Python implementation before source can"
        " be generated."
    )
  try:
    parsed = ast.parse(implementation)
  except SyntaxError as error:
    raise ValueError(
        f"Tool {node.label!r} has invalid Python: {error.msg} (line"
        f" {error.lineno})."
    ) from error
  if not any(
      isinstance(statement, (ast.AsyncFunctionDef, ast.FunctionDef))
      and statement.name == identifier
      for statement in parsed.body
  ):
    raise ValueError(
        f"Tool implementation must define a function named {identifier!r}."
    )
  return [implementation.rstrip(), ""]


def _identifier(node: GraphNode) -> str:
  if not node.label.isidentifier() or node.label == "user":
    raise ValueError(
        f"Node name {node.label!r} must be a Python identifier other than"
        " 'user'."
    )
  return node.label
