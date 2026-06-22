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

import{$a as s,Bb as r,Ca as m,Cb as u,Cc as n,Ib as d,Kb as c,Pa as l,Yb as v,Zb as o,_b as g,ac as f,eb as p,pd as b,wc as h}from"./chunk-2SRK2U7X.js";import"./chunk-RMXJBC7V.js";var M=["a2ui-slider",""],E=(()=>{class a extends b{value=n.required();label=n("");minValue=n.required();maxValue=n.required();inputId=super.getUniqueId("a2ui-slider");resolvedValue=h(()=>super.resolvePrimitive(this.value())??0);handleInput(t){let i=this.value()?.path;!(t.target instanceof HTMLInputElement)||!i||this.processor.setData(this.component(),i,t.target.valueAsNumber,this.surfaceId())}static \u0275fac=(()=>{let t;return function(e){return(t||(t=m(a)))(e||a)}})();static \u0275cmp=s({type:a,selectors:[["","a2ui-slider",""]],inputs:{value:[1,"value"],label:[1,"label"],minValue:[1,"minValue"],maxValue:[1,"maxValue"]},features:[p],attrs:M,decls:4,vars:14,consts:[[3,"for"],["autocomplete","off","type","range",3,"input","value","min","max","id"]],template:function(i,e){i&1&&(r(0,"section")(1,"label",0),g(2),u(),r(3,"input",1),c("input",function(y){return e.handleInput(y)}),u()()),i&2&&(o(e.theme.components.Slider.container),l(),o(e.theme.components.Slider.label),d("htmlFor",e.inputId),l(),f(" ",e.label()," "),l(),v(e.theme.additionalStyles==null?null:e.theme.additionalStyles.Slider),o(e.theme.components.Slider.element),d("value",e.resolvedValue())("min",e.minValue())("max",e.maxValue())("id",e.inputId))},styles:["[_nghost-%COMP%]{display:block;flex:var(--weight)}input[_ngcontent-%COMP%]{display:block;width:100%;box-sizing:border-box}"]})}return a})();export{E as Slider};
