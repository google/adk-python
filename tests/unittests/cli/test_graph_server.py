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

import ast
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.cli.graph._graph_document import generate_source
from google.adk.cli.graph._graph_document import GraphDocument
from google.adk.cli.graph.graph_server import GraphServer
from google.adk.cli.graph.inspector import GraphEdge
from google.adk.cli.graph.inspector import GraphNode
from google.adk.cli.graph.inspector import GraphTopology
import pytest


def test_graph_draft_updates_the_topology_visible_to_the_workbench() -> None:
  """Saving a draft makes newly added canvas nodes available on reload."""
  server = GraphServer(topology=GraphTopology(root_id="root"))
  client = TestClient(server.app)
  draft = GraphTopology(
      root_id="researcher",
      nodes=[
          GraphNode(
              id="researcher",
              type="llm_agent",
              label="Researcher",
          )
      ],
  )

  document = GraphDocument(
      topology=draft,
      positions={"researcher": (240.0, 120.0)},
  )

  response = client.put(
      "/api/graph/topology",
      json={"document": document.model_dump(), "expected_revision": 0},
  )

  assert response.status_code == 200
  assert response.json()["revision"] == 1
  assert response.json()["positions"] == {"researcher": [240.0, 120.0]}
  assert (
      client.get("/api/graph/topology").json()["topology"] == draft.model_dump()
  )


def test_graph_server_defaults_to_loopback() -> None:
  """The source-writing workbench is never network-exposed by default."""
  server = GraphServer(topology=GraphTopology(root_id=""))

  assert server.host == "127.0.0.1"


def test_standalone_graph_exposes_read_only_code() -> None:
  """A graph without an agent source disables code editing in the workbench."""
  client = TestClient(GraphServer(topology=None).app)

  response = client.get("/api/graph/code")

  assert response.status_code == 200
  assert response.json()["editable"] is False


def test_workbench_exposes_smart_layout_and_accessible_icon_tooltips() -> None:
  """The canvas sizes labels, groups relationships, and labels icon controls."""
  client = TestClient(GraphServer(topology=None).app)

  page = client.get("/").text

  assert "wrapText(node.label,24)" in page
  assert "levelSeparation:190" in page
  assert "function advancedLayout()" in page
  assert "function highlightRelatedNodes(id)" in page
  assert "function validateDraft()" in page
  assert "function inspectEdge(id)" in page
  assert 'id="node-instruction"' in page
  assert 'id="reconnect-edge"' in page
  assert "smooth:{type:'horizontal'" in page
  assert 'id="layout" class="btn">Auto layout' in page
  assert 'id="add-node" class="btn primary">+ Add' in page
  assert "function toolsFor(node)" in page
  assert "node.type!=='tool'" in page
  assert "function updateNewNodeFields()" in page
  assert 'id="new-tool-implementation"' in page
  assert "Select an LLM agent before adding a tool." in page
  assert "split(/\\s+/)" in page
  assert "function undo()" in page
  assert "function redo()" in page
  assert "let graphDocument" in page
  assert "let document =" not in page
  assert "font color" not in page


def test_graph_rejects_a_stale_draft_save() -> None:
  """Concurrent browser saves cannot silently overwrite each other."""
  client = TestClient(GraphServer(topology=GraphTopology(root_id="")).app)
  document = GraphDocument(topology=GraphTopology(root_id=""))

  first_response = client.put(
      "/api/graph/topology",
      json={"document": document.model_dump(), "expected_revision": 0},
  )
  stale_response = client.put(
      "/api/graph/topology",
      json={"document": document.model_dump(), "expected_revision": 0},
  )

  assert first_response.status_code == 200
  assert stale_response.status_code == 409


def test_graph_rejects_tool_bound_to_a_non_llm_agent() -> None:
  """Tools cannot be connected to workflow containers or other tools."""
  topology = GraphTopology(
      root_id="pipeline",
      nodes=[
          GraphNode(id="pipeline", type="sequential", label="pipeline"),
          GraphNode(id="search", type="tool", label="search"),
      ],
      edges=[
          GraphEdge(
              id="bad-tool-edge",
              source="search",
              target="pipeline",
              type="tool_binding",
          )
      ],
  )
  client = TestClient(GraphServer(topology=GraphTopology(root_id="")).app)

  response = client.put(
      "/api/graph/topology",
      json={
          "document": GraphDocument(topology=topology).model_dump(),
          "expected_revision": 0,
      },
  )

  assert response.status_code == 422
  assert "tool connection" in response.json()["detail"].lower()


def test_generated_source_is_valid_python_for_a_connected_agent_graph() -> None:
  """A managed graph produces source with agents, children, and tools wired."""
  topology = GraphTopology(
      root_id="coordinator",
      nodes=[
          GraphNode(
              id="coordinator",
              type="llm_agent",
              label="coordinator",
              config={"instruction": "Delegate research."},
          ),
          GraphNode(
              id="researcher",
              type="llm_agent",
              label="researcher",
              config={"model": "gemini-2.5-flash"},
          ),
          GraphNode(
              id="search_web",
              type="tool",
              label="search_web",
              description="Searches the web.",
              config={
                  "implementation": (
                      "def search_web(query: str) -> str:\n"
                      "    return f'Results for {query}'"
                  )
              },
          ),
      ],
      edges=[
          GraphEdge(
              id="child",
              source="coordinator",
              target="researcher",
              type="sub_agent",
          ),
          GraphEdge(
              id="tool",
              source="search_web",
              target="researcher",
              type="tool_binding",
          ),
      ],
  )

  source = generate_source(topology)

  ast.parse(source)
  assert "sub_agents=[researcher]" in source
  assert "tools=[search_web]" in source
  assert "root_agent = coordinator" in source


def test_generated_source_rejects_tools_without_an_implementation() -> None:
  """Source generation never replaces a real tool with a nonfunctional stub."""
  topology = GraphTopology(
      root_id="agent",
      nodes=[
          GraphNode(id="agent", type="llm_agent", label="agent"),
          GraphNode(id="lookup", type="tool", label="lookup"),
      ],
      edges=[
          GraphEdge(
              id="tool", source="lookup", target="agent", type="tool_binding"
          )
      ],
  )

  with pytest.raises(ValueError, match="needs a Python implementation"):
    generate_source(topology)


def test_generated_source_rejects_non_python_node_names() -> None:
  """Generated source cannot contain an invalid or reserved ADK agent name."""
  topology = GraphTopology(
      root_id="invalid name",
      nodes=[
          GraphNode(id="invalid name", type="llm_agent", label="invalid name")
      ],
  )

  with pytest.raises(ValueError, match="Python identifier"):
    generate_source(topology)


def test_generated_container_source_constructs_the_adk_agent_tree() -> None:
  """Sequential containers are generated in child-first construction order."""
  topology = GraphTopology(
      root_id="pipeline",
      nodes=[
          GraphNode(id="pipeline", type="sequential", label="pipeline"),
          GraphNode(id="writer", type="llm_agent", label="writer"),
      ],
      edges=[
          GraphEdge(
              id="child", source="pipeline", target="writer", type="sub_agent"
          )
      ],
  )
  namespace: dict[str, object] = {}

  exec(compile(generate_source(topology), "<generated>", "exec"), namespace)

  assert namespace["root_agent"].name == "pipeline"
  assert namespace["root_agent"].sub_agents[0].name == "writer"


def test_applying_a_graph_writes_valid_source_and_preserves_the_draft(
    tmp_path: Path,
) -> None:
  """Applying a managed graph atomically writes its generated ADK source."""
  agent_file = tmp_path / "agent.py"
  agent_file.write_text("# original source\n", encoding="utf-8")
  topology = GraphTopology(
      root_id="researcher",
      nodes=[
          GraphNode(
              id="researcher",
              type="llm_agent",
              label="researcher",
              config={"instruction": "Research the supplied question."},
          )
      ],
  )
  server = GraphServer(topology=topology, agent_file_path=agent_file)
  client = TestClient(server.app)

  response = client.post("/api/graph/code/apply")

  assert response.status_code == 200
  assert agent_file.read_text(encoding="utf-8") == response.json()["code"]
  ast.parse(response.json()["code"])
  persisted = GraphServer(
      topology=GraphTopology(root_id=""), agent_file_path=agent_file
  )
  assert persisted.document.topology == topology


def test_preview_shows_generated_source_without_modifying_the_agent_file(
    tmp_path: Path,
) -> None:
  """Previewing a graph does not overwrite source before an explicit apply."""
  agent_file = tmp_path / "agent.py"
  agent_file.write_text("# original source\n", encoding="utf-8")
  topology = GraphTopology(
      root_id="writer",
      nodes=[GraphNode(id="writer", type="llm_agent", label="writer")],
  )
  client = TestClient(
      GraphServer(topology=topology, agent_file_path=agent_file).app
  )

  response = client.post("/api/graph/code/preview")

  assert response.status_code == 200
  assert "root_agent = writer" in response.json()["code"]
  assert agent_file.read_text(encoding="utf-8") == "# original source\n"


def test_empty_persisted_draft_does_not_hide_an_inspected_graph(
    tmp_path: Path,
) -> None:
  """A partial blank draft recovers to the authoritative inspected graph."""
  agent_file = tmp_path / "agent.py"
  agent_file.write_text("root_agent = None\n", encoding="utf-8")
  topology = GraphTopology(
      root_id="writer",
      nodes=[GraphNode(id="writer", type="llm_agent", label="writer")],
  )
  server = GraphServer(topology=topology, agent_file_path=agent_file)
  server._document_store.save(
      document=GraphDocument(topology=GraphTopology(root_id="")),
      expected_revision=0,
  )

  recovered = GraphServer(topology=topology, agent_file_path=agent_file)

  assert recovered.document.topology == topology


def test_invalid_manual_source_is_not_written(tmp_path: Path) -> None:
  """Saving malformed Python preserves the last working source file."""
  agent_file = tmp_path / "agent.py"
  agent_file.write_text("root_agent = None\n", encoding="utf-8")
  client = TestClient(
      GraphServer(
          topology=GraphTopology(root_id=""), agent_file_path=agent_file
      ).app
  )

  response = client.post("/api/graph/code", json={"code": "def broken(:\n"})

  assert response.status_code == 422
  assert agent_file.read_text(encoding="utf-8") == "root_agent = None\n"


def test_manual_source_save_creates_a_backup_and_requires_graph_reload(
    tmp_path: Path,
) -> None:
  """Manual source changes preserve recovery data and mark topology stale."""
  agent_file = tmp_path / "agent.py"
  agent_file.write_text("root_agent = None\n", encoding="utf-8")
  client = TestClient(
      GraphServer(
          topology=GraphTopology(root_id=""), agent_file_path=agent_file
      ).app
  )

  response = client.post("/api/graph/code", json={"code": "root_agent = 1\n"})

  assert response.status_code == 200
  assert response.json()["requires_reload"] is True
  assert agent_file.read_text(encoding="utf-8") == "root_agent = 1\n"
  assert (tmp_path / ".adk" / "graph-backups" / "agent.py").read_text(
      encoding="utf-8"
  ) == "root_agent = None\n"
