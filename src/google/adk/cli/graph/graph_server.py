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
import tempfile
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from ._graph_document import generate_source
from ._graph_document import GraphDocument
from ._graph_document import GraphDocumentConflictError
from ._graph_document import GraphDocumentStore
from ._graph_document import validate_topology
from .inspector import GraphTopology

logger = logging.getLogger("google_adk." + __name__)


def _atomic_write(path: Path, contents: str) -> None:
  """Atomically replaces a UTF-8 source file after caller-side validation."""
  if path.is_file():
    backup_path = path.parent / ".adk" / "graph-backups" / path.name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(path.read_bytes())
  with tempfile.NamedTemporaryFile(
      mode="w", encoding="utf-8", dir=path.parent, delete=False
  ) as temporary_file:
    temporary_file.write(contents)
    temporary_path = Path(temporary_file.name)
  temporary_path.replace(path)


class CodeSaveRequest(BaseModel):
  code: str


class GraphDraftRequest(BaseModel):
  """A revision-checked graph document update from the workbench."""

  document: GraphDocument
  expected_revision: int


class GraphServer:
  """Serves an editable visual graph workbench for ADK agents."""

  def __init__(
      self,
      topology: Optional[GraphTopology],
      agent_file_path: Optional[Path] = None,
      host: str = "127.0.0.1",
      port: int = 8000,
  ):
    self.agent_file_path = agent_file_path
    self.host = host
    self.port = port
    initial_topology = (
        topology.model_copy(deep=True)
        if topology
        else GraphTopology(root_id="")
    )
    self._document_store = GraphDocumentStore(agent_file_path=agent_file_path)
    self.document = self._document_store.load(topology=initial_topology)
    self.app = FastAPI(title="ADK Graph Workbench")
    self._setup_routes()

  def _setup_routes(self) -> None:
    @self.app.get("/api/graph/topology")
    def get_topology() -> dict:
      return self.document.model_dump()

    @self.app.put("/api/graph/topology")
    def save_topology(req: GraphDraftRequest) -> dict:
      try:
        if req.expected_revision != self.document.revision:
          raise GraphDocumentConflictError(
              "This graph has changed in another browser. Reload before saving."
          )
        validate_topology(req.document.topology)
        self.document = self._document_store.save(
            document=req.document,
            expected_revision=req.expected_revision,
        )
      except GraphDocumentConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
      except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
      return self.document.model_dump()

    @self.app.get("/api/graph/code")
    def get_code() -> dict:
      if not self.agent_file_path or not self.agent_file_path.exists():
        return {
            "code": "# Standalone builder mode - no file selected",
            "editable": False,
        }
      return {
          "code": self.agent_file_path.read_text(encoding="utf-8"),
          "editable": True,
      }

    @self.app.post("/api/graph/code")
    def save_code(req: CodeSaveRequest) -> dict:
      if not self.agent_file_path:
        raise HTTPException(
            status_code=400, detail="No source file attached to save code."
        )
      try:
        compile(req.code, str(self.agent_file_path), "exec")
      except SyntaxError as error:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Source contains invalid Python: {error.msg} (line"
                f" {error.lineno})."
            ),
        ) from error
      _atomic_write(self.agent_file_path, req.code)
      return {
          "status": "saved",
          "path": str(self.agent_file_path),
          "requires_reload": True,
      }

    @self.app.post("/api/graph/code/preview")
    def preview_generated_code() -> dict:
      try:
        return {"code": generate_source(self.document.topology)}
      except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    @self.app.post("/api/graph/code/apply")
    def apply_generated_code() -> dict:
      if not self.agent_file_path:
        raise HTTPException(
            status_code=400, detail="No source file attached to save code."
        )
      try:
        source = generate_source(self.document.topology)
      except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
      _atomic_write(self.agent_file_path, source)
      self.document = self._document_store.update_source_digest(
          document=self.document
      )
      return {
          "status": "saved",
          "path": str(self.agent_file_path),
          "code": source,
      }

    @self.app.get("/", response_class=HTMLResponse)
    def index() -> str:
      return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADK Graph Workbench</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root { --blue:#1a73e8; --ink:#202124; --muted:#5f6368; --line:#dadce0; --surface:#fff; --canvas:#f8fafd; --focus:#8ab4f8; }
    * { box-sizing:border-box; }
    body { margin:0; min-width:320px; font:14px/1.4 Arial,Roboto,sans-serif; color:var(--ink); background:var(--canvas); height:100vh; overflow:hidden; }
    button,input,select,textarea { font:inherit; }
    button { cursor:pointer; }
    button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
    .topbar { height:64px; display:flex; align-items:center; gap:16px; padding:0 20px; background:var(--surface); border-bottom:1px solid var(--line); }
    .mark { display:flex; gap:3px; } .mark i { width:9px; height:9px; border-radius:50%; } .mark i:nth-child(1){background:#4285f4}.mark i:nth-child(2){background:#ea4335}.mark i:nth-child(3){background:#fbbc04}.mark i:nth-child(4){background:#34a853}
    .brand { font-size:18px; font-weight:500; white-space:nowrap; } .crumb { color:var(--muted); border-left:1px solid var(--line); padding-left:16px; }
    .spacer { flex:1; } .status { color:var(--muted); font-size:12px; } .status.dirty { color:#b06000; font-weight:600; }
    .btn { min-height:38px; border:1px solid var(--line); border-radius:20px; padding:0 15px; color:#1967d2; background:var(--surface); font-weight:600; }
    .btn:hover { background:#f1f6fe; } .btn.primary { border-color:var(--blue); background:var(--blue); color:#fff; } .btn.primary:hover { background:#185abc; }
    .icon-btn { width:40px; height:40px; padding:0; border:0; border-radius:50%; background:transparent; color:var(--muted); font-size:20px; } .icon-btn:hover { background:#f1f3f4; }
    .workspace { height:calc(100vh - 64px); display:grid; grid-template-columns:272px minmax(0,1fr) 312px; transition:grid-template-columns .2s ease; }
    .workspace.left-closed { grid-template-columns:0 minmax(0,1fr) 312px; } .workspace.right-closed { grid-template-columns:272px minmax(0,1fr) 0; } .workspace.left-closed.right-closed { grid-template-columns:0 minmax(0,1fr) 0; }
    .rail,.inspector { background:var(--surface); overflow:auto; border-right:1px solid var(--line); } .inspector { border-right:0; border-left:1px solid var(--line); }
    .rail-inner,.inspector-inner { padding:18px; min-width:272px; } .section-title { color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; margin:4px 0 12px; }
    .hint { color:var(--muted); font-size:12px; margin:0 0 16px; } .palette { display:grid; gap:8px; }
    .palette-item { display:flex; align-items:center; gap:10px; width:100%; padding:11px; text-align:left; border:1px solid var(--line); border-radius:12px; background:#fff; } .palette-item:hover { border-color:#a8c7fa; background:#f8fbff; box-shadow:0 1px 2px #00000014; }
    .type-dot { width:10px; height:10px; border-radius:3px; background:var(--type,#1a73e8); } .palette-item small { display:block; color:var(--muted); margin-top:1px; }
    .canvas-wrap { min-width:0; position:relative; overflow:hidden; background-color:var(--canvas); background-image:radial-gradient(#dfe5ed 1px,transparent 1px); background-size:20px 20px; }
    #network { width:100%; height:100%; } .canvas-tools { position:absolute; z-index:2; top:16px; left:16px; display:flex; gap:8px; padding:6px; border:1px solid var(--line); border-radius:12px; background:#fff; box-shadow:0 1px 3px #00000020; }
    .canvas-tools .icon-btn { width:34px; height:34px; font-size:17px; position:relative; } .canvas-tools .icon-btn.active { color:var(--blue); background:#e8f0fe; }
    [data-tooltip]::after { content:attr(data-tooltip); position:absolute; z-index:30; left:50%; top:calc(100% + 8px); width:max-content; max-width:220px; padding:6px 9px; border-radius:6px; background:#303134; color:#fff; font:12px/1.3 Arial,Roboto,sans-serif; box-shadow:0 2px 6px #0004; opacity:0; pointer-events:none; transform:translate(-50%,-3px); transition:opacity .15s ease,transform .15s ease; } [data-tooltip]:hover::after,[data-tooltip]:focus-visible::after { opacity:1; transform:translate(-50%,0); }
    .empty { position:absolute; inset:0; display:none; place-items:center; pointer-events:none; } .empty-card { width:360px; text-align:center; padding:28px; background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 2px 6px #00000016; } .empty-card h2 { margin:0 0 8px; font-size:20px; } .empty-card p { margin:0; color:var(--muted); }
    .field { margin-bottom:16px; } .field label { display:block; margin:0 0 6px; font-size:12px; font-weight:700; color:var(--muted); } .field input,.field select,.field textarea { width:100%; padding:10px; border:1px solid var(--line); border-radius:8px; background:#fff; } .field textarea { min-height:86px; resize:vertical; }
    .node-title { font-size:18px; font-weight:500; margin:0 0 4px; } .node-meta { color:var(--muted); margin:0 0 18px; } .divider { height:1px; background:var(--line); margin:18px 0; } .connection-list { margin:0; padding-left:18px; color:var(--muted); font-size:13px; }
    .drawer { position:fixed; z-index:5; bottom:0; left:0; right:0; height:0; background:#fff; border-top:0 solid var(--line); box-shadow:0 -6px 18px #0000001c; overflow:hidden; transition:height .22s ease,border .22s ease; } .drawer.open { height:min(47vh,470px); border-top-width:1px; }
    .drawer-head { height:54px; padding:0 20px; display:flex; align-items:center; gap:12px; border-bottom:1px solid var(--line); } .drawer-body { height:calc(100% - 54px); padding:14px 20px; } #code { width:100%; height:100%; resize:none; padding:14px; border:1px solid var(--line); border-radius:8px; font:13px/1.55 "Roboto Mono",Consolas,monospace; }
    .modal-backdrop { position:fixed; z-index:10; inset:0; display:none; place-items:center; background:#20212488; } .modal-backdrop.open { display:grid; } .modal { width:min(460px,calc(100vw - 32px)); padding:24px; border-radius:16px; background:#fff; box-shadow:0 12px 28px #0000004d; } .modal h2 { margin:0 0 20px; font-size:20px; } .modal-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:20px; }
    .toast { position:fixed; z-index:20; left:50%; bottom:24px; transform:translateX(-50%) translateY(90px); padding:11px 16px; border-radius:8px; color:#fff; background:#303134; box-shadow:0 2px 8px #0004; transition:transform .2s; } .toast.show { transform:translateX(-50%) translateY(0); }
    @media (max-width:900px) { .workspace,.workspace.left-closed,.workspace.right-closed { grid-template-columns:0 minmax(0,1fr) 0; } .rail,.inspector { position:fixed; z-index:4; top:64px; bottom:0; width:290px; box-shadow:2px 0 10px #0002; transform:translateX(-100%); transition:transform .2s; } .inspector { right:0; transform:translateX(100%); box-shadow:-2px 0 10px #0002; } .workspace.mobile-left .rail,.workspace.mobile-right .inspector { transform:translateX(0); } .crumb { display:none; } .status { display:none; } }
  </style>
</head>
<body>
  <header class="topbar"><span class="mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="brand">Google ADK</span><span class="crumb">Graph workbench</span><span class="spacer"></span><span id="status" class="status" aria-live="polite">Loading graph…</span><button class="btn" id="code-toggle" title="Open code drawer">&lt;/&gt; Code</button><button class="btn primary" id="save-draft">Save draft</button></header>
  <main id="workspace" class="workspace">
    <aside class="rail" aria-label="Node library"><div class="rail-inner"><div class="section-title">Build</div><p class="hint">Drag a card onto the canvas, or select it to add a node.</p><div id="palette" class="palette"></div><div class="divider"></div><button id="close-left" class="btn">Hide library</button></div></aside>
    <section class="canvas-wrap" aria-label="Agent graph canvas"><div class="canvas-tools"><button id="toggle-left" class="icon-btn" aria-label="Toggle node library" data-tooltip="Node library">☰</button><button id="undo" class="icon-btn" aria-label="Undo last graph change" data-tooltip="Undo (Ctrl/Cmd+Z)">↶</button><button id="redo" class="icon-btn" aria-label="Redo graph change" data-tooltip="Redo (Ctrl/Cmd+Shift+Z)">↷</button><button id="connect" class="icon-btn" aria-label="Connect nodes" data-tooltip="Connect nodes">↗</button><button id="layout" class="icon-btn" aria-label="Auto-arrange graph" data-tooltip="Smart auto-arrange">⌘</button><button id="fit" class="icon-btn" aria-label="Fit graph to canvas" data-tooltip="Fit to canvas (F)">⊙</button><button id="zoom-in" class="icon-btn" aria-label="Zoom in" data-tooltip="Zoom in">+</button><button id="zoom-out" class="icon-btn" aria-label="Zoom out" data-tooltip="Zoom out">−</button><button id="toggle-right" class="icon-btn" aria-label="Toggle inspector" data-tooltip="Inspector">ⓘ</button></div><div id="network"></div><div id="empty" class="empty"><div class="empty-card"><h2>Start building your agent system</h2><p>Add an agent or tool from the library. Your source code stays tucked away until you open the Code drawer.</p></div></div></section>
    <aside class="inspector" aria-label="Inspector"><div class="inspector-inner"><div class="section-title">Inspector</div><div id="inspector-empty" class="hint">Select a node or connection to edit its details.</div><form id="node-form" hidden><h2 id="node-heading" class="node-title"></h2><p id="node-type" class="node-meta"></p><div class="field"><label for="node-label">Name</label><input id="node-label" required></div><div class="field"><label for="node-description">Description</label><textarea id="node-description"></textarea></div><div class="field"><label for="node-kind">Node type</label><select id="node-kind"><option value="llm_agent">LLM agent</option><option value="sequential">Sequential workflow</option><option value="parallel">Parallel workflow</option><option value="loop">Loop workflow</option><option value="tool">Tool</option></select></div><button class="btn primary" type="submit">Apply changes</button><button id="delete-node" class="btn" type="button">Delete node</button></form><div id="edge-inspector" hidden><h2 class="node-title">Connection</h2><p id="edge-type" class="node-meta"></p><button id="delete-edge" class="btn" type="button">Delete connection</button></div><div class="divider"></div><button id="close-right" class="btn">Hide inspector</button></div></aside>
  </main>
  <section id="drawer" class="drawer" aria-label="Source code"><div class="drawer-head"><strong>Python source</strong><span id="code-status" class="status"></span><span class="spacer"></span><button id="preview-code" class="btn">Preview generated code</button><button id="apply-code" class="btn primary">Apply graph to source</button><button id="save-code" class="btn">Save code</button><button id="close-code" class="icon-btn" aria-label="Close code drawer" data-tooltip="Close code drawer">×</button></div><div class="drawer-body"><textarea id="code" spellcheck="false"></textarea></div></section>
  <div id="modal-backdrop" class="modal-backdrop" role="presentation"><form id="node-modal" class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title"><h2 id="modal-title">Add node</h2><div class="field"><label for="new-name">Name</label><input id="new-name" placeholder="Researcher" required autocomplete="off"></div><div class="field"><label for="new-type">Type</label><select id="new-type"><option value="llm_agent">LLM agent</option><option value="sequential">Sequential workflow</option><option value="parallel">Parallel workflow</option><option value="loop">Loop workflow</option><option value="tool">Tool</option></select></div><div class="field"><label for="new-description">Description</label><textarea id="new-description" placeholder="What does this component do?"></textarea></div><div class="modal-actions"><button id="cancel-modal" class="btn" type="button">Cancel</button><button class="btn primary" type="submit">Add to canvas</button></div></form></div><div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    const TYPES = { llm_agent:['Agent','#1a73e8'], sequential:['Sequential','#7b1fa2'], parallel:['Parallel','#00897b'], loop:['Loop','#e37400'], tool:['Tool','#d93025'] };
    let graphDocument = {revision:0,topology:{root_id:'',nodes:[],edges:[]},positions:{}}, topology = graphDocument.topology, network, nodes, edges, selected = null, sourceEditable = false, history = [], future = [];
    const $ = id => document.getElementById(id); const status = (text, dirty=false) => { $('status').textContent=text; $('status').classList.toggle('dirty',dirty); };
    function toast(message) { $('toast').textContent=message; $('toast').classList.add('show'); setTimeout(() => $('toast').classList.remove('show'), 2500); }
    function wrapText(text, limit) { const words=String(text||'').trim().split(/\s+/).filter(Boolean); const lines=[]; let line=''; words.forEach(word=>{ const next=line ? line+' '+word : word; if(next.length>limit && line){lines.push(line);line=word;}else{line=next;} }); if(line)lines.push(line); return lines; }
    function nodeView(node) { const [typeName,color] = TYPES[node.type] || ['Component','#5f6368']; const summary=(node.description||node.config?.instruction||'').replace(/\s+/g,' ').trim(); const lines=[...wrapText(node.label,24),typeName,...wrapText(summary,34).slice(0,2)]; const longest=Math.max(...lines.map(line=>line.length),12); return { id:node.id, label:lines.join('\n'), width:Math.max(158,Math.min(310,longest*7+42)), shape:'box', margin:{top:13,right:18,bottom:13,left:18}, color:{background:'#ffffff',border:color,highlight:{background:'#e8f0fe',border:'#1a73e8'}}, borderWidth:2, chosen:{node:(values)=>{values.shadow=true;values.shadowColor='#1a73e844';}}, font:{face:'Arial',color:'#202124',size:14,align:'center'}, shadow:true }; }
    function edgeView(edge) { const tool=edge.type==='tool_binding'; const labels={sub_agent:'sub-agent',tool_binding:'tool',app_plugin:'app plugin',workflow_contains:'contains',workflow_route:'route'}; return { id:edge.id, from:edge.source, to:edge.target, arrows:'to', dashes:tool, color:{color:tool?'#d93025':'#5f6368',highlight:'#1a73e8'}, width:2, smooth:{type:'cubicBezier',roundness:.25}, font:{color:'#5f6368',size:11,align:'middle'}, label:edge.label||labels[edge.type]||edge.type }; }
    function graphLayout() { return { enabled:true, direction:'UD', sortMethod:'directed', nodeSpacing:250, treeSpacing:280, levelSeparation:190, blockShifting:true, edgeMinimization:true, parentCentralization:true, shakeTowards:'roots' }; }
    function nodeSize(node) { const summary=(node.description||node.config?.instruction||'').replace(/\s+/g,' ').trim(); const lines=[...wrapText(node.label,24),...(TYPES[node.type]||['Component']).slice(0,1),...wrapText(summary,34).slice(0,2)]; return {width:Math.max(158,Math.min(310,Math.max(...lines.map(line=>line.length),12)*7+42)),height:Math.max(78,lines.length*20+28)}; }
    function advancedLayout() { const byId=new Map(topology.nodes.map(node=>[node.id,node])); const children=new Map(topology.nodes.map(node=>[node.id,[]])); topology.edges.filter(edge=>edge.type==='sub_agent').forEach(edge=>children.get(edge.source)?.push(edge.target)); const depth=new Map(); const visit=(id,level)=>{ if(!byId.has(id)||level<=(depth.get(id)??-1))return; depth.set(id,level); children.get(id)?.forEach(child=>visit(child,level+1)); }; if(topology.root_id)visit(topology.root_id,0); topology.nodes.filter(node=>node.type!=='tool'&&!depth.has(node.id)).forEach(node=>visit(node.id,0)); const levels=[]; depth.forEach((level,id)=>{(levels[level]??=[]).push(id);}); const positions={}; let maxWidth=0, y=0; levels.forEach(ids=>{ const sizes=ids.map(id=>nodeSize(byId.get(id))); const rowHeight=Math.max(...sizes.map(size=>size.height),80); const rowWidth=sizes.reduce((total,size)=>total+size.width,0)+Math.max(0,ids.length-1)*72; let x=-rowWidth/2; ids.forEach((id,index)=>{const size=sizes[index];positions[id]=[x+size.width/2,y];x+=size.width+72;}); maxWidth=Math.max(maxWidth,rowWidth); y+=rowHeight+128; }); const tools=topology.nodes.filter(node=>node.type==='tool'); let toolY=0; tools.forEach(tool=>{ const bindings=topology.edges.filter(edge=>edge.type==='tool_binding'&&edge.source===tool.id); const targets=bindings.map(edge=>positions[edge.target]).filter(Boolean); const preferredY=targets.length?targets.reduce((total,position)=>total+position[1],0)/targets.length:toolY; const size=nodeSize(tool); positions[tool.id]=[-maxWidth/2-size.width/2-120,Math.max(toolY,preferredY)]; toolY=positions[tool.id][1]+size.height+48; }); return positions; }
    function positionedNode(node) { const position=graphDocument.positions[node.id]; return position ? {...nodeView(node),x:position[0],y:position[1]} : nodeView(node); }
    function capturePositions(nodeIds) { const positions=network.getPositions(nodeIds); Object.entries(positions).forEach(([id,position])=>{graphDocument.positions[id]=[position.x,position.y];}); }
    function clearGraphFocus() { topology.nodes.forEach(node=>nodes.update(positionedNode(node))); topology.edges.forEach(edge=>edges.update(edgeView(edge))); }
    function highlightRelatedNodes(id) { const node=topology.nodes.find(item=>item.id===id); if(!node)return; clearGraphFocus(); const matchingTools=node.type==='tool'?topology.nodes.filter(item=>item.type==='tool'&&item.label===node.label).map(item=>item.id):[id]; const highlighted=new Set(matchingTools); const highlightedEdges=new Set(); topology.edges.forEach(edge=>{if(matchingTools.includes(edge.source)||edge.source===id||edge.target===id){highlighted.add(edge.source);highlighted.add(edge.target);highlightedEdges.add(edge.id);}}); topology.nodes.forEach(item=>{const view=positionedNode(item);const isMatch=highlighted.has(item.id);nodes.update({...view,color:{background:isMatch?'#e8f0fe':'#fff',border:isMatch?'#1a73e8':'#dadce0'},borderWidth:isMatch?3:1,font:{...view.font,color:isMatch?'#202124':'#80868b'},shadow:isMatch});}); topology.edges.forEach(edge=>{const view=edgeView(edge);edges.update({...view,color:{...view.color,color:highlightedEdges.has(edge.id)?'#1a73e8':'#c4c7c5'},width:highlightedEdges.has(edge.id)?3:1});}); if(node.type==='tool'&&matchingTools.length>1)toast(`${matchingTools.length} instances of ${node.label} highlighted`); }
    function render() { nodes = new vis.DataSet(topology.nodes.map(positionedNode)); edges = new vis.DataSet(topology.edges.map(edgeView)); if (network) network.destroy(); const hasPositions=Object.keys(graphDocument.positions).length>0; network = new vis.Network($('network'), {nodes,edges}, { physics:false, layout:{hierarchical:hasPositions?false:graphLayout()}, interaction:{dragNodes:true,dragView:true,hover:true,multiselect:true,navigationButtons:false}, manipulation:{enabled:false,addEdge:(data,callback)=>{ if(data.from===data.to){toast('A node cannot connect to itself.'); callback(null);return;} const source=topology.nodes.find(n=>n.id===data.from); const type=source && source.type==='tool' ? 'tool_binding' : 'sub_agent'; topology.edges.push({id:'edge_'+Date.now(),source:data.from,target:data.to,type,label:null}); callback(null); $('connect').classList.remove('active'); render(); markDirty('Connection added'); }} }); network.on('selectNode', params => inspectNode(params.nodes[0])); network.on('selectEdge', params => inspectEdge(params.edges[0])); network.on('deselectNode', () => {clearGraphFocus();clearInspector();}); network.on('dragEnd', params => { capturePositions(params.nodes); markDirty('Layout changed'); }); $('empty').style.display=topology.nodes.length?'none':'grid'; }
    function populatePalette() { $('palette').innerHTML=Object.entries(TYPES).map(([type,[name,color]])=>`<button class="palette-item" draggable="true" data-type="${type}" style="--type:${color}"><span class="type-dot"></span><span><strong>${name}</strong><small>Add ${name.toLowerCase()}</small></span></button>`).join(''); document.querySelectorAll('.palette-item').forEach(item=>{ item.addEventListener('click',()=>openAdd(item.dataset.type)); item.addEventListener('dragstart', event=>event.dataTransfer.setData('text/plain',item.dataset.type)); }); }
    function openAdd(type='llm_agent') { $('new-type').value=type; $('new-name').value=''; $('new-description').value=''; $('modal-backdrop').classList.add('open'); $('new-name').focus(); }
    function closeAdd() { $('modal-backdrop').classList.remove('open'); }
    function snapshot() { return JSON.stringify({topology,positions:graphDocument.positions}); }
    function updateHistoryButtons() { $('undo').disabled=history.length<2; $('redo').disabled=!future.length; }
    function markDirty(message='Draft changed') { const current=snapshot(); if(history.at(-1)!==current){history.push(current);history=history.slice(-100);future=[];} updateHistoryButtons(); status(message,true); }
    function restoreSnapshot(saved) { const restored=JSON.parse(saved); topology=restored.topology; graphDocument.topology=topology; graphDocument.positions=restored.positions||{}; clearInspector(); render(); status('Draft changed',true); updateHistoryButtons(); }
    function undo() { if(history.length<2){toast('Nothing to undo.');return;} future.push(history.pop()); restoreSnapshot(history.at(-1)); }
    function redo() { if(!future.length){toast('Nothing to redo.');return;} const saved=future.pop(); history.push(saved); restoreSnapshot(saved); }
    function selectedNode() { return topology.nodes.find(node=>node.id===selected); }
    function inspectNode(id) { selected=id; highlightRelatedNodes(id); const node=selectedNode(); $('node-form').hidden=false; $('edge-inspector').hidden=true; $('inspector-empty').hidden=true; $('node-heading').textContent=node.label; $('node-type').textContent=(TYPES[node.type]||['Component'])[0]; $('node-label').value=node.label; $('node-description').value=node.description||node.config?.instruction||''; $('node-kind').value=node.type; }
    function inspectEdge(id) { selected=id; const edge=topology.edges.find(item=>item.id===id); $('node-form').hidden=true; $('edge-inspector').hidden=false; $('inspector-empty').hidden=true; $('edge-type').textContent=edge.type==='tool_binding'?'Tool binding':'Sub-agent connection'; }
    function clearInspector() { selected=null; $('node-form').hidden=true; $('edge-inspector').hidden=true; $('inspector-empty').hidden=false; }
    async function saveDraft() { graphDocument.topology=topology; const response=await fetch('/api/graph/topology',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({document:graphDocument,expected_revision:graphDocument.revision})}); const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Unable to save draft'); graphDocument=data; topology=graphDocument.topology; status('Draft saved'); toast('Graph draft saved'); }
    async function saveCode() { if(!sourceEditable){ toast('No source file is attached to this graph.'); return; } const response=await fetch('/api/graph/code',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:$('code').value})}); const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Unable to save code'); $('code-status').textContent=data.requires_reload?'Saved — restart graph to reload topology':'Saved'; toast(data.requires_reload?'Source saved. Restart the graph server to reload it.':'Python source saved'); }
    async function previewGeneratedCode() { const response=await fetch('/api/graph/code/preview',{method:'POST'}); const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Unable to generate code'); $('code').value=data.code; $('code-status').textContent='Generated preview — review before applying'; codeDrawer(true); }
    async function applyGeneratedCode() { if(!sourceEditable){ toast('No source file is attached to this graph.'); return; } if(!confirm('Replace the current source file with code generated from this graph?')) return; const response=await fetch('/api/graph/code/apply',{method:'POST'}); const data=await response.json(); if(!response.ok) throw new Error(data.detail||'Unable to apply graph'); $('code').value=data.code; $('code-status').textContent='Generated code applied'; toast('Generated ADK source saved'); }
    $('node-modal').addEventListener('submit', event=>{ event.preventDefault(); const label=$('new-name').value.trim(); if(!label) return; const id=label.replace(/[^A-Za-z0-9_]/g,'_')+'_'+Date.now(); topology.nodes.push({id,type:$('new-type').value,label,description:$('new-description').value.trim(),sub_agents:[],tools:[],parent_id:null,config:{}}); if(!topology.root_id) topology.root_id=id; closeAdd(); render(); network.selectNodes([id]); markDirty('Node added'); });
    $('node-form').addEventListener('submit', event=>{ event.preventDefault(); const node=selectedNode(); if(!node) return; node.label=$('node-label').value.trim()||node.label; node.description=$('node-description').value.trim(); node.type=$('node-kind').value; render(); network.selectNodes([node.id]); markDirty('Node updated'); });
    $('delete-node').onclick=()=>{ const node=selectedNode(); if(!node||!confirm(`Delete ${node.label}?`))return; topology.nodes=topology.nodes.filter(item=>item.id!==node.id); topology.edges=topology.edges.filter(edge=>edge.source!==node.id&&edge.target!==node.id); if(topology.root_id===node.id) topology.root_id=topology.nodes[0]?.id||''; clearInspector(); render(); markDirty('Node deleted'); };
    $('delete-edge').onclick=()=>{ topology.edges=topology.edges.filter(edge=>edge.id!==selected); clearInspector(); render(); markDirty('Connection deleted'); };
    $('save-draft').onclick=()=>saveDraft().catch(error=>toast(error.message)); $('save-code').onclick=()=>saveCode().catch(error=>toast(error.message)); $('preview-code').onclick=()=>previewGeneratedCode().catch(error=>toast(error.message)); $('apply-code').onclick=()=>applyGeneratedCode().catch(error=>toast(error.message));
    $('undo').onclick=undo; $('redo').onclick=redo; $('connect').onclick=()=>{ network.addEdgeMode(); $('connect').classList.add('active'); toast('Drag from a source node to a target node.'); }; $('layout').onclick=()=>{ graphDocument.positions=advancedLayout(); render(); network.fit({animation:{duration:260,easingFunction:'easeInOutQuad'}}); markDirty('Smart layout applied'); toast('Grouped related agents and spaced nodes by label size.'); }; $('fit').onclick=()=>network.fit({animation:true}); $('zoom-in').onclick=()=>network.moveTo({scale:network.getScale()*1.15}); $('zoom-out').onclick=()=>network.moveTo({scale:network.getScale()*.85});
    const toggle = (side,force) => { const mobile=innerWidth<=900; $('workspace').classList.toggle(mobile?'mobile-'+side:side+'-closed',force); }; $('toggle-left').onclick=()=>toggle('left'); $('close-left').onclick=()=>toggle('left',true); $('toggle-right').onclick=()=>toggle('right'); $('close-right').onclick=()=>toggle('right',true);
    const codeDrawer=open=>{ $('drawer').classList.toggle('open',open); $('code-toggle').classList.toggle('primary',open); }; $('code-toggle').onclick=()=>codeDrawer(!$('drawer').classList.contains('open')); $('close-code').onclick=()=>codeDrawer(false); $('cancel-modal').onclick=closeAdd;
    $('network').addEventListener('dragover',event=>event.preventDefault()); $('network').addEventListener('drop',event=>{ event.preventDefault(); openAdd(event.dataTransfer.getData('text/plain')||'llm_agent'); });
    document.addEventListener('keydown',event=>{ if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();saveDraft().catch(error=>toast(error.message));} if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();} if(event.key==='Escape'){closeAdd();network?.disableEditMode();$('connect').classList.remove('active');} if(event.key.toLowerCase()==='a'&&document.activeElement===document.body)openAdd(); if(event.key.toLowerCase()==='f'&&document.activeElement===document.body)network?.fit({animation:true}); });
    async function initialize() { try { populatePalette(); const [graph,code]=await Promise.all([fetch('/api/graph/topology'),fetch('/api/graph/code')]); if(!graph.ok)throw new Error('Unable to load graph'); graphDocument=await graph.json(); graphDocument.positions=graphDocument.positions||{}; topology=graphDocument.topology; history=[snapshot()]; updateHistoryButtons(); const source=await code.json(); $('code').value=source.code; sourceEditable=source.editable; $('save-code').disabled=!sourceEditable; $('apply-code').disabled=!sourceEditable; $('code-status').textContent=sourceEditable?'':'Read-only: no file attached'; render(); status('Draft loaded'); } catch(error) { status('Unable to load graph'); toast(error.message); } }
    initialize();
  </script>
</body>
</html>
      """

  def run(self) -> None:
    uvicorn.run(self.app, host=self.host, port=self.port)
