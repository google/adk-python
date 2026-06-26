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

import{a as s,b as u,c as i,d as n,e as m,f as t,g as o,h as l,o as d,q as h}from"./chunk-NALL4A3P.js";var A=class extends h{static{t(this,"ArchitectureTokenBuilder")}constructor(){super(["architecture"])}},C=class extends d{static{t(this,"ArchitectureValueConverter")}runCustomConverter(c,r,a){if(c.name==="ARCH_ICON")return r.replace(/[()]/g,"").trim();if(c.name==="ARCH_TEXT_ICON")return r.replace(/["()]/g,"");if(c.name==="ARCH_TITLE"){let e=r.replace(/^\[|]$/g,"").trim();return(e.startsWith('"')&&e.endsWith('"')||e.startsWith("'")&&e.endsWith("'"))&&(e=e.slice(1,-1),e=e.replace(/\\"/g,'"').replace(/\\'/g,"'")),e.trim()}}},v={parser:{TokenBuilder:t(()=>new A,"TokenBuilder"),ValueConverter:t(()=>new C,"ValueConverter")}};function f(c=n){let r=i(u(c),o),a=i(s({shared:r}),l,v);return r.ServiceRegistry.register(a),{shared:r,Architecture:a}}t(f,"createArchitectureServices");export{v as a,f as b};
