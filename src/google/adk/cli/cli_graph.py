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

import logging
from pathlib import Path
from typing import Optional

import click

from .utils.agent_loader import AgentLoader
from .graph.inspector import AgentInspector
from .graph.graph_server import GraphServer

logger = logging.getLogger("google_adk." + __name__)


@click.command("graph")
@click.argument("agent_file", type=click.Path(exists=True), required=False)
@click.option("--host", default="0.0.0.0", help="Host address to bind the web server.")
@click.option("--port", default=8000, type=int, help="Port to serve the visual graph UI.")
def graph_cmd(agent_file: Optional[str], host: str, port: int) -> None:
  """Inspect and visualize agent topology interactively."""
  if not agent_file:
    click.echo("Starting ADK Graph Server in standalone builder mode...")
    server = GraphServer(topology=None, host=host, port=port)
    server.run()
    return

  click.echo(f"Inspecting agent at: {agent_file}")
  path = Path(agent_file)
  agent_or_app = AgentLoader.load_agent_or_app(path)

  inspector = AgentInspector(agent_or_app)
  topology = inspector.inspect()

  click.echo(f"Successfully parsed agent graph! Total nodes: {len(topology.nodes)}, edges: {len(topology.edges)}")
  click.echo(f"Serving Visual Agent Graph on http://{host}:{port}")

  server = GraphServer(topology=topology, agent_file_path=path, host=host, port=port)
  server.run()
