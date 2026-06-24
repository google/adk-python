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

import{a as o,b as n,c as i,d as s,e as m,f as e,g as u,l as d,o as c,q as l}from"./chunk-NALL4A3P.js";var v=class extends l{static{e(this,"PieTokenBuilder")}constructor(){super(["pie","showData"])}},C=class extends c{static{e(this,"PieValueConverter")}runCustomConverter(t,r,a){if(t.name==="PIE_SECTION_LABEL")return r.replace(/"/g,"").trim()}},P={parser:{TokenBuilder:e(()=>new v,"TokenBuilder"),ValueConverter:e(()=>new C,"ValueConverter")}};function p(t=s){let r=i(n(t),u),a=i(o({shared:r}),d,P);return r.ServiceRegistry.register(a),{shared:r,Pie:a}}e(p,"createPieServices");export{P as a,p as b};
