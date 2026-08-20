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
  """FastAPI server serving Google Material 3 UI visual agent graph canvas and code editor."""

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
    self.app = FastAPI(title="ADK Visual Graph Workbench")
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
        <title>Google ADK Visual Graph Workbench</title>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; font-family: 'Google Sans', -apple-system, sans-serif; background: #202124; color: #e8eaed; height: 100vh; display: flex; flex-direction: column; }
          header { padding: 12px 24px; background: #2d2e31; border-bottom: 1px solid #3c4043; display: flex; justify-content: space-between; align-items: center; }
          .logo { display: flex; align-items: center; gap: 10px; font-weight: 500; font-size: 1.1rem; color: #ffffff; }
          .logo svg { fill: #8ab4f8; }
          .toolbar { display: flex; gap: 10px; align-items: center; }
          .container { display: flex; flex: 1; overflow: hidden; }
          .panel { flex: 1; padding: 16px; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #3c4043; }
          .panel:last-child { border-right: none; }
          .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
          .panel-title { font-weight: 500; font-size: 0.85rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.5px; }
          #mynetwork { flex: 1; background: #171717; border: 1px solid #3c4043; border-radius: 12px; overflow: hidden; }
          textarea { flex: 1; background: #171717; color: #e8eaed; border: 1px solid #3c4043; border-radius: 12px; padding: 16px; font-family: 'Roboto Mono', monospace; font-size: 13px; resize: none; outline: none; line-height: 1.6; }
          textarea:focus { border-color: #8ab4f8; }
          
          /* Google Material Design Buttons */
          .btn { background: #8ab4f8; color: #202124; border: none; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 500; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: background 0.2s; }
          .btn:hover { background: #aecbfa; }
          .btn-secondary { background: #3c4043; color: #e8eaed; }
          .btn-secondary:hover { background: #4d5156; }
          #toast { color: #81c995; font-size: 0.8rem; margin-left: 10px; display: none; font-weight: 500; }
          
          /* Google Material Modal Overlay */
          .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.6); display: none; justify-content: center; align-items: center; z-index: 1000; backdrop-filter: blur(4px); }
          .modal { background: #2d2e31; border: 1px solid #3c4043; border-radius: 16px; width: 440px; padding: 24px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 12px 32px rgba(0,0,0,0.4); }
          .modal h3 { margin: 0; color: #ffffff; font-size: 1.2rem; font-weight: 500; }
          .form-group { display: flex; flex-direction: column; gap: 6px; }
          .form-group label { font-size: 0.8rem; color: #9aa0a6; font-weight: 500; }
          .form-group input, .form-group select, .form-group textarea { background: #171717; border: 1px solid #3c4043; color: #e8eaed; padding: 10px 14px; border-radius: 8px; font-size: 13px; font-family: inherit; outline: none; }
          .form-group input:focus, .form-group select:focus, .form-group textarea:focus { border-color: #8ab4f8; }
          .modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px; }
        </style>
      </head>
      <body>
        <header>
          <div class="logo">
            <svg width="24" height="24" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14.5v-9l6 4.5-6 4.5z"/></svg>
            <span>Google ADK Visual Graph Workbench</span>
          </div>
          <div class="toolbar">
            <button class="btn" onclick="openAddNodeModal()">+ Add Agent / Tool</button>
          </div>
        </header>
        <div class="container">
          <div class="panel">
            <div class="panel-header">
              <span class="panel-title">Visual Multi-Agent Canvas</span>
              <span style="font-size: 0.75rem; color: #9aa0a6;">Hierarchical Layout | Drag to Rearrange</span>
            </div>
            <div id="mynetwork"></div>
          </div>
          <div class="panel">
            <div class="panel-header">
              <span class="panel-title">Agent Source Code (Python)</span>
              <div>
                <button class="btn btn-secondary" onclick="saveCode()">Save & Update Graph</button>
                <span id="toast">Saved & Synced!</span>
              </div>
            </div>
            <textarea id="code-editor" placeholder="# ADK Python code..."></textarea>
          </div>
        </div>

        <!-- Node Add/Edit Modal -->
        <div id="nodeModal" class="modal-overlay">
          <div class="modal">
            <h3 id="modalTitle">Add Agent or Tool</h3>
            <div class="form-group">
              <label>Node Name (Python Variable)</label>
              <input type="text" id="nodeName" placeholder="e.g. ResearcherAgent">
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
              <textarea id="nodeInstruction" rows="3" placeholder="e.g. Gather research data from web"></textarea>
            </div>
            <div class="modal-actions">
              <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
              <button class="btn" onclick="submitNodeModal()">Add Node</button>
            </div>
          </div>
        </div>

        <script>
          let network = null;
          let currentTopology = null;

          function colorForType(type) {
            switch(type) {
              case 'sequential': return { background: '#1a73e8', border: '#8ab4f8' };
              case 'parallel': return { background: '#9334e6', border: '#c58af9' };
              case 'loop': return { background: '#e37400', border: '#fde293' };
              case 'llm_agent': return { background: '#137333', border: '#81c995' };
              case 'tool': return { background: '#c5221f', border: '#f28b82' };
              default: return { background: '#3c4043', border: '#9aa0a6' };
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
                label: `${iconForType(node.type)} ${node.label}\\n[${node.type}]`,
                shape: 'box',
                margin: 14,
                borderRadius: 8,
                color: {
                  background: colors.background,
                  border: colors.border,
                  highlight: { background: colors.border, border: '#ffffff' }
                },
                font: { color: '#ffffff', face: 'Google Sans, sans-serif', size: 13, bold: true }
              });
            });

            topology.edges.forEach(edge => {
              const isTool = edge.type === 'tool_binding';
              edgesArray.push({
                from: edge.source,
                to: edge.target,
                label: edge.type === 'sub_agent' ? 'sub-agent' : 'tool',
                arrows: { to: { enabled: true, scaleFactor: 0.8 } },
                dashes: isTool,
                smooth: { type: 'cubicBezier', roundness: 0.5 },
                color: { color: isTool ? '#f28b82' : '#8ab4f8', highlight: '#ffffff' },
                font: { color: '#9aa0a6', size: 11, face: 'Google Sans', align: 'middle' }
              });
            });

            const container = document.getElementById('mynetwork');
            const data = {
              nodes: new vis.DataSet(nodesArray),
              edges: new vis.DataSet(edgesArray)
            };

            const options = {
              layout: {
                hierarchical: {
                  enabled: true,
                  direction: 'UD',
                  sortMethod: 'directed',
                  nodeSpacing: 160,
                  levelSeparation: 120
                }
              },
              physics: { enabled: false },
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
            document.getElementById('modalTitle').innerText = "Add Agent or Tool";
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
