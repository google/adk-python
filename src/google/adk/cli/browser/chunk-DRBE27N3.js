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

import{k as m}from"./chunk-WBLSVR3V.js";import{Y as s,r as p}from"./chunk-QFMJV7VH.js";import{g as a}from"./chunk-JRNAXTJ7.js";import{j as g}from"./chunk-RMXJBC7V.js";var S=a(({flowchart:o})=>{let n=o?.subGraphTitleMargin?.top??0,t=o?.subGraphTitleMargin?.bottom??0,r=n+t;return{subGraphTitleTopMargin:n,subGraphTitleBottomMargin:t,subGraphTitleTotalMargin:r}},"getSubGraphTitleMargins");function b(o,n){return g(this,null,function*(){let t=o.getElementsByTagName("img");if(!t||t.length===0)return;let r=n.replace(/<img[^>]*>/g,"").trim()==="";yield Promise.all([...t].map(e=>new Promise(u=>{function i(){if(e.style.display="flex",e.style.flexDirection="column",r){let f=s().fontSize?s().fontSize:window.getComputedStyle(document.body).fontSize,c=5,[d=p.fontSize]=m(f),l=d*c+"px";e.style.minWidth=l,e.style.maxWidth=l}else e.style.width="100%";u(e)}a(i,"setupImage"),setTimeout(()=>{e.complete&&i()}),e.addEventListener("error",i),e.addEventListener("load",i)})))})}a(b,"configureLabelImages");export{S as a,b};
