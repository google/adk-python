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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from .inspector import GraphTopology

logger = logging.getLogger("google_adk." + __name__)


class CodeSaveRequest(BaseModel):
  code: str


class GraphServer:
  """FastAPI server serving interactive visual graph canvas with node editing & code synchronization."""

  def __init__(
      self,
      topology: Optional[GraphTopology],
      agent_file_path: Optional[Path] = None,
      host: str = "0.0.0.0",
      port: int = 8000,
  ):
    self.topology = topology
    self.agent_file_path = agent_file_path
    self.host = host
    self.port = port
    self.app = FastAPI(title="ADK Interactive Graph Canvas & Code Workbench")
    self._setup_routes()

  def _setup_routes(self) -> None:
    @self.app.get("/api/graph/topology")
    def get_topology():
      if self.topology is None:
        return {"nodes": [], "edges": [], "root_id": ""}
      return self.topology.model_dump()

    @self.app.get("/api/graph/code")
    def get_code():
      if not self.agent_file_path or not self.agent_file_path.exists():
        return {"code": "# Standalone builder mode - no file selected"}
      return {"code": self.agent_file_path.read_text(encoding="utf-8")}

    @self.app.post("/api/graph/code")
    def save_code(req: CodeSaveRequest):
      if not self.agent_file_path:
        raise HTTPException(status_code=400, detail="No source file attached to save code.")
      self.agent_file_path.write_text(req.code, encoding="utf-8")
      return {"status": "saved", "path": str(self.agent_file_path)}

    @self.app.get("/", response_class=HTMLResponse)
    def index():
      return """
      <!DOCTYPE html>
      <html>
      <head>
        <title>ADK Visual Graph Builder & Code Editor</title>
        <meta charset="utf-8">
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }
          header { padding: 12px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
          h1 { margin: 0; font-size: 1.1rem; color: #38bdf8; }
          .toolbar { display: flex; gap: 8px; align-items: center; }
          .container { display: flex; flex: 1; overflow: hidden; }
          .panel { flex: 1; padding: 16px; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #334155; }
          .panel:last-child { border-right: none; }
          .panel-title { font-weight: bold; margin-bottom: 12px; font-size: 0.9rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
          #mynetwork { flex: 1; background: #020617; border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
          textarea { flex: 1; background: #020617; color: #38bdf8; border: 1px solid #334155; border-radius: 8px; padding: 16px; font-family: 'Fira Code', monospace; font-size: 13px; resize: none; outline: none; line-height: 1.5; }
          button { background: #0284c7; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-size: 12px; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 4px; }
          button:hover { background: #0369a1; }
          button.btn-success { background: #059669; }
          button.btn-success:hover { background: #047857; }
          #toast { color: #4ade80; font-size: 0.8rem; margin-left: 10px; display: none; }
          
          /* Modal Popup for Node Editing & Creation */
          .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: none; justify-content: center; align-items: center; z-index: 1000; }
          .modal { background: #1e293b; border: 1px solid #475569; border-radius: 8px; width: 420px; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
          .modal h3 { margin: 0; color: #38bdf8; font-size: 1.1rem; }
          .form-group { display: flex; flex-direction: column; gap: 6px; }
          .form-group label { font-size: 0.8rem; color: #94a3b8; font-weight: 600; }
          .form-group input, .form-group select, .form-group textarea { background: #0f172a; border: 1px solid #334155; color: white; padding: 8px 12px; border-radius: 4px; font-size: 13px; outline: none; }
          .modal-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px; }
        </style>
      </head>
      <body>
        <header>
          <h1>ADK Visual Graph Builder & Code Workbench</h1>
          <div class="toolbar">
            <button class="btn-success" onclick="openAddNodeModal()">+ Add Node (Agent / Tool)</button>
          </div>
        </header>
        <div class="container">
          <div class="panel">
            <div class="panel-title">
              <span>VISUAL AGENT CANVAS</span>
              <span style="font-size: 0.8rem; color: #64748b;">Double-click node to edit properties | Drag to connect</span>
            </div>
            <div id="mynetwork"></div>
          </div>
          <div class="panel">
            <div class="panel-title">
              <span>AGENT SOURCE CODE (Python)</span>
              <div>
                <button onclick="saveCode()">Save & Re-Parse Graph</button>
                <span id="toast">Saved & Updated!</span>
              </div>
            </div>
            <textarea id="code-editor" placeholder="# Python code editor..."></textarea>
          </div>
        </div>

        <!-- Node Add/Edit Modal -->
        <div id="nodeModal" class="modal-overlay">
          <div class="modal">
            <h3 id="modalTitle">Add Agent or Tool</h3>
            <div class="form-group">
              <label>Node Name (Variable Identifier)</label>
              <input type="text" id="nodeName" placeholder="e.g. ValidatorAgent">
            </div>
            <div class="form-group">
              <label>Node Type</label>
              <select id="nodeType">
                <option value="llm_agent">🤖 LLM Agent (LlmAgent)</option>
                <option value="sequential">🔄 Sequential Container (SequentialAgent)</option>
                <option value="parallel">⚡ Parallel Container (ParallelAgent)</option>
                <option value="loop">🔁 Loop Container (LoopAgent)</option>
                <option value="tool">🛠️ Function Tool (Tool)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Instruction / Description</label>
              <textarea id="nodeInstruction" rows="3" placeholder="e.g. Validate output dataset"></textarea>
            </div>
            <div class="modal-actions">
              <button style="background: #475569;" onclick="closeModal()">Cancel</button>
              <button class="btn-success" onclick="submitNodeModal()">Add to Graph & Code</button>
            </div>
          </div>
        </div>

        <script>
          let network = null;
          let currentTopology = null;

          function colorForType(type) {
            switch(type) {
              case 'sequential': return { background: '#0284c7', border: '#38bdf8' };
              case 'parallel': return { background: '#7c3aed', border: '#a78bfa' };
              case 'loop': return { background: '#d97706', border: '#fbbf24' };
              case 'llm_agent': return { background: '#059669', border: '#34d399' };
              case 'tool': return { background: '#dc2626', border: '#f87171' };
              default: return { background: '#475569', border: '#94a3b8' };
            }
          }

          function iconForType(type) {
            switch(type) {
              case 'sequential': return '🔄 ';
              case 'parallel': return '⚡ ';
              case 'loop': return '🔁 ';
              case 'llm_agent': return '🤖 ';
              case 'tool': return '🛠️ ';
              default: return '📦 ';
            }
          }

          function renderGraph(topology) {
            if (!topology || !topology.nodes) return;
            currentTopology = topology;

            const nodesArray = [];
            const edgesArray = [];

            topology.nodes.forEach(node => {
              const colors = colorForType(node.type);
              nodesArray.push({
                id: node.id,
                label: `${iconForType(node.type)}${node.label}\\n[${node.type}]`,
                shape: 'box',
                margin: 12,
                color: {
                  background: colors.background,
                  border: colors.border,
                  highlight: { background: colors.border, border: '#ffffff' }
                },
                font: { color: '#ffffff', face: 'monospace', size: 14 }
              });
            });

            topology.edges.forEach(edge => {
              const isTool = edge.type === 'tool_binding';
              edgesArray.push({
                from: edge.source,
                to: edge.target,
                label: edge.type,
                arrows: 'to',
                dashes: isTool,
                color: { color: isTool ? '#f87171' : '#38bdf8', highlight: '#ffffff' },
                font: { color: '#94a3b8', size: 10, align: 'middle' }
              });
            });

            const container = document.getElementById('mynetwork');
            const data = {
              nodes: new vis.DataSet(nodesArray),
              edges: new vis.DataSet(edgesArray)
            };

            const options = {
              physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01, springLength: 100 }
              },
              interaction: { hover: true, dragNodes: true, zoomView: true, dragView: true }
            };

            if (network) network.destroy();
            network = new vis.Network(container, data, options);

            network.on("doubleClick", function (params) {
              if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const targetNode = currentTopology.nodes.find(n => n.id === nodeId);
                if (targetNode) {
                  openEditNodeModal(targetNode);
                }
              }
            });
          }

          fetch('/api/graph/topology')
            .then(res => res.json())
            .then(data => renderGraph(data));

          fetch('/api/graph/code')
            .then(res => res.json())
            .then(data => {
              document.getElementById('code-editor').value = data.code;
            });

          function saveCode() {
            const code = document.getElementById('code-editor').value;
            fetch('/api/graph/code', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ code })
            })
            .then(res => res.json())
            .then(data => {
              const toast = document.getElementById('toast');
              toast.style.display = 'inline';
              setTimeout(() => { toast.style.display = 'none'; }, 2000);

              fetch('/api/graph/topology')
                .then(res => res.json())
                .then(topo => renderGraph(topo));
            });
          }

          function openAddNodeModal() {
            document.getElementById('modalTitle').innerText = "Add New Agent or Tool";
            document.getElementById('nodeName').value = "";
            document.getElementById('nodeInstruction').value = "";
            document.getElementById('nodeModal').style.display = "flex";
          }

          function closeModal() {
            document.getElementById('nodeModal').style.display = "none";
          }

          function submitNodeModal() {
            const name = document.getElementById('nodeName').value.trim();
            const type = document.getElementById('nodeType').value;
            const instruction = document.getElementById('nodeInstruction').value.trim();

            if (!name) {
              alert("Please enter a valid node name.");
              return;
            }

            let codeSnippet = "";
            if (type === "llm_agent") {
              codeSnippet = "\\n# New LLM Agent\\n" + name + " = LlmAgent(\\n    name=\\"" + name + "\\",\\n    instruction=\\"" + instruction + "\\",\\n)\\n";
            } else if (type === "sequential") {
              codeSnippet = "\\n# New Sequential Pipeline\\n" + name + " = SequentialAgent(\\n    name=\\"" + name + "\\",\\n    description=\\"" + instruction + "\\",\\n    sub_agents=[],\\n)\\n";
            } else if (type === "parallel") {
              codeSnippet = "\\n# New Parallel Container\\n" + name + " = ParallelAgent(\\n    name=\\"" + name + "\\",\\n    description=\\"" + instruction + "\\",\\n    sub_agents=[],\\n)\\n";
            } else if (type === "tool") {
              codeSnippet = "\\ndef " + name + "(query: str) -> str:\\n    \\"\\"\\"" + instruction + "\\"\\"\\"\\n    return f\\"Processed {query}\\"\\n";
            }

            const editor = document.getElementById('code-editor');
            editor.value += codeSnippet;
            closeModal();
            saveCode();
          }

          function openEditNodeModal(node) {
            document.getElementById('modalTitle').innerText = "Edit " + node.label;
            document.getElementById('nodeName').value = node.label;
            document.getElementById('nodeType').value = node.type;
            document.getElementById('nodeInstruction').value = node.config.instruction || node.description || "";
            document.getElementById('nodeModal').style.display = "flex";
          }
        </script>
      </body>
      </html>
      """

  def run(self):
    uvicorn.run(self.app, host=self.host, port=self.port)
