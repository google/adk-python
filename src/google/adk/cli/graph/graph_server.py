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
  """FastAPI server serving visual graph topology, real-time execution events, and split-screen code editing."""

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
    self.app = FastAPI(title="ADK Graph Visualizer & Debugger")
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
        <title>ADK Visual Graph & Code Editor</title>
        <meta charset="utf-8">
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }
          header { padding: 12px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
          h1 { margin: 0; font-size: 1.1rem; color: #38bdf8; }
          .container { display: flex; flex: 1; overflow: hidden; }
          .panel { flex: 1; padding: 16px; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #334155; }
          .panel:last-child { border-right: none; }
          .panel-title { font-weight: bold; margin-bottom: 8px; font-size: 0.9rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
          textarea { flex: 1; background: #020617; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; padding: 12px; font-family: monospace; font-size: 13px; resize: none; outline: none; }
          pre { flex: 1; background: #020617; color: #a7f3d0; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin: 0; overflow: auto; font-size: 12px; }
          button { background: #0284c7; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-size: 12px; cursor: pointer; font-weight: 600; }
          button:hover { background: #0369a1; }
          #toast { color: #4ade80; font-size: 0.8rem; margin-left: 10px; display: none; }
        </style>
      </head>
      <body>
        <header>
          <h1>ADK Visual Graph & Live Code Workbench</h1>
          <div id="status" style="font-size: 0.85rem; color: #94a3b8;">Loaded</div>
        </header>
        <div class="container">
          <div class="panel">
            <div class="panel-title">
              <span>VISUAL GRAPH TOPOLOGY</span>
            </div>
            <pre id="graph-json">Loading topology...</pre>
          </div>
          <div class="panel">
            <div class="panel-title">
              <span>AGENT SOURCE CODE (Python)</span>
              <div>
                <button onclick="saveCode()">Save Code</button>
                <span id="toast">Saved!</span>
              </div>
            </div>
            <textarea id="code-editor" placeholder="# Python code editor..."></textarea>
          </div>
        </div>
        <script>
          fetch('/api/graph/topology')
            .then(res => res.json())
            .then(data => {
              document.getElementById('graph-json').innerText = JSON.stringify(data, null, 2);
            });

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
            });
          }
        </script>
      </body>
      </html>
      """

  def run(self):
    uvicorn.run(self.app, host=self.host, port=self.port)
