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

import{N as w}from"./chunk-QFMJV7VH.js";import{g as r,i as c}from"./chunk-JRNAXTJ7.js";var g=r((t,e,i,h)=>{t.attr("class",i);let{width:o,height:n,x,y:u}=s(t,e);w(t,n,o,h);let a=m(x,u,o,n,e);t.attr("viewBox",a),c.debug(`viewBox configured: ${a} with padding: ${e}`)},"setupViewPortForSVG"),s=r((t,e)=>{let i=t.node()?.getBBox()||{width:0,height:0,x:0,y:0};return{width:i.width+e*2,height:i.height+e*2,x:i.x,y:i.y}},"calculateDimensionsWithPadding"),m=r((t,e,i,h,o)=>`${t-o} ${e-o} ${i} ${h}`,"createViewBox");export{g as a};
