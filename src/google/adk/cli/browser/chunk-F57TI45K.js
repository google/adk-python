// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import{b as y,c as f,d as p,e as h}from"./chunk-TNJPXCAB.js";import{a as g,d,g as u}from"./chunk-W7PRDKNL.js";import{c as m}from"./chunk-WBLSVR3V.js";import{A as l,M as s}from"./chunk-QFMJV7VH.js";import{g as o,i}from"./chunk-JRNAXTJ7.js";import{j as a}from"./chunk-RMXJBC7V.js";var L={common:s,getConfig:l,insertCluster:d,insertEdge:p,insertEdgeLabel:y,insertMarkers:h,insertNode:u,interpolateToCurve:m,labelHelper:g,log:i,positionEdgeLabel:f},t={},w=o(r=>{for(let e of r)t[e.name]=e},"registerLayoutLoaders"),c=o(()=>{w([{name:"dagre",loader:o(()=>a(null,null,function*(){return yield import("./chunk-TNYN2TVW.js")}),"loader")},{name:"cose-bilkent",loader:o(()=>a(null,null,function*(){return yield import("./chunk-HD4LLD2O.js")}),"loader")}])},"registerDefaultLayoutLoaders");c();var C=o((r,e)=>a(null,null,function*(){if(!(r.layoutAlgorithm in t))throw new Error(`Unknown layout algorithm: ${r.layoutAlgorithm}`);let n=t[r.layoutAlgorithm];return(yield n.loader()).render(r,e,L,{algorithm:n.algorithm})}),"render"),D=o((r="",{fallback:e="dagre"}={})=>{if(r in t)return r;if(e in t)return i.warn(`Layout algorithm ${r} is not registered. Using ${e} as fallback.`),e;throw new Error(`Both layout algorithms ${r} and ${e} are not registered.`)},"getRegisteredLayoutAlgorithm");export{w as a,C as b,D as c};
